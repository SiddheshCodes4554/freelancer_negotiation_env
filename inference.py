"""OpenEnv hackathon inference runner with strict stdout format."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")
MAX_STEPS = 5
TASKS = ["easy", "medium", "hard"]
ENV_NAME = "freelancer-negotiation"


def _required_token() -> str:
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return HF_TOKEN


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _safe_text(value: Any) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _build_prompt(task: str, observation: dict[str, Any], step: int) -> str:
    return (
        "You are a freelancer negotiating with a client. "
        "Return ONLY valid JSON with keys message and action_type. "
        "action_type must be one of: negotiate, accept, reject.\n"
        f"task={task}\n"
        f"step={step}\n"
        f"observation={json.dumps(observation, ensure_ascii=True)}"
    )


def _parse_action(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    payload: dict[str, Any]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {
                "message": "I can deliver this with clear scope and timeline for Rs 1200.",
                "action_type": "negotiate",
            }
        payload = json.loads(text[start : end + 1])

    action_type = str(payload.get("action_type", "negotiate")).strip().lower()
    if action_type not in {"negotiate", "accept", "reject"}:
        action_type = "negotiate"

    message = str(payload.get("message", "I can deliver this with clear scope and timeline for Rs 1200.")).strip()
    if not message:
        message = "I can deliver this with clear scope and timeline for Rs 1200."

    return {"message": message, "action_type": action_type}


def _llm_action(client: OpenAI, task: str, observation: dict[str, Any], step: int) -> dict[str, str]:
    prompt = _build_prompt(task=task, observation=observation, step=step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            top_p=1,
            messages=[
                {
                    "role": "system",
                    "content": "Return only JSON with keys message and action_type.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        return _parse_action(content)
    except Exception:
        return {
            "message": "I can deliver this with clear scope and timeline for Rs 1200.",
            "action_type": "negotiate",
        }


def _print_start(task: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def _print_step(step: int, action_str: str, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action={_safe_text(action_str)} reward={reward:.2f} "
        f"done={_bool_text(done)} error=null",
        flush=True,
    )


def _print_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} rewards={rewards_str}", flush=True)


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{OPENENV_BASE_URL.rstrip('/')}/{path.lstrip('/')}",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response for {path}")
    return data


def run_task(client: OpenAI, task: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    success = False

    _print_start(task)

    try:
        reset_data = _post_json("reset", {})
        observation = reset_data.get("observation", {})
        if not isinstance(observation, dict):
            observation = {}

        done = bool(reset_data.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = _llm_action(client=client, task=task, observation=observation, step=step)
            action_payload = {"action": action}

            step_data = _post_json("step", action_payload)
            reward = float(step_data.get("reward", 0.0) or 0.0)
            done = bool(step_data.get("done", False))
            observation = step_data.get("observation", {})
            if not isinstance(observation, dict):
                observation = {}

            action_str = f"{action['action_type']}:{action['message']}"
            _print_step(step=step, action_str=action_str, reward=reward, done=done)

            rewards.append(reward)
            steps_taken = step

        success = bool(done)
    except Exception:
        success = False
    finally:
        _print_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    token = _required_token()
    client = OpenAI(base_url=API_BASE_URL, api_key=token)

    for task in TASKS:
        run_task(client=client, task=task)


if __name__ == "__main__":
    main()
