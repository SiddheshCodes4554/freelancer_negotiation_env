# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task definitions and deterministic graders for freelancer negotiation.

This module provides:
- Three benchmark tasks (easy, medium, hard) with initial state and expected outcome.
- Deterministic graders that return a score in [0.0, 1.0].

Each grader evaluates:
- Final price quality
- Decision quality (accept/reject behavior)
- Negotiation quality (message quality, efficiency, and scenario handling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


TaskDifficulty = Literal["easy", "medium", "hard"]
DecisionType = Literal["accept", "reject", "negotiate"]


@dataclass(frozen=True)
class TaskDefinition:
    """Benchmark task specification for the negotiation environment."""

    task_id: str
    difficulty: TaskDifficulty
    title: str
    description: str
    initial_state: dict[str, object]
    expected_outcome: dict[str, object]


@dataclass
class EpisodeResult:
    """Deterministic grading input extracted from an episode run."""

    final_price: float | None
    decision: DecisionType
    conversation_history: list[str]
    step_count: int
    client_type: str


TASKS: dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        difficulty="easy",
        title="High Budget Easy Deal",
        description="Client has healthy budget and expects a quick professional agreement.",
        initial_state={
            "current_price": 1400.0,
            "client_budget": 2200.0,
            "deadline": "2026-04-30",
            "conversation_history": [
                "client: I have a healthy budget and want to close quickly.",
                "client: Share a fair quote and timeline and we can proceed.",
            ],
            "client_type": "premium",
            "revisions": 1,
        },
        expected_outcome={
            "target_decision": "accept",
            "ideal_price": 1500.0,
            "max_steps_for_efficiency": 3,
        },
    ),
    "medium": TaskDefinition(
        task_id="medium",
        difficulty="medium",
        title="Tight Budget Negotiation",
        description="Budget is constrained, so the agent must negotiate terms effectively.",
        initial_state={
            "current_price": 1350.0,
            "client_budget": 1050.0,
            "deadline": "2026-04-10",
            "conversation_history": [
                "client: My budget is tight, but I still need a quality delivery.",
                "client: Can we negotiate scope and price to make this work?",
            ],
            "client_type": "normal",
            "revisions": 2,
        },
        expected_outcome={
            "target_decision": "negotiate",
            "ideal_price": 1200.0,
            "max_steps_for_efficiency": 5,
        },
    ),
    "hard": TaskDefinition(
        task_id="hard",
        difficulty="hard",
        title="Low Budget High Expectations",
        description="Client asks for premium outcomes with unrealistic budget and scope demands.",
        initial_state={
            "current_price": 1800.0,
            "client_budget": 700.0,
            "deadline": "2026-04-08",
            "conversation_history": [
                "client: I need premium quality fast, but my budget is very low.",
                "client: I also expect extra changes without increasing budget.",
            ],
            "client_type": "toxic",
            "revisions": 3,
        },
        expected_outcome={
            "target_decision": "reject",
            "ideal_price": 1700.0,
            "min_viable_price": 1450.0,
            "max_steps_for_efficiency": 5,
            "boundary_keywords": ["scope", "paid", "revision", "contract", "milestone"],
        },
    ),
}


# Keep a practical margin away from exact 0/1 so downstream rounding in
# external validators cannot collapse strict-range scores to boundary values.
_SCORE_EPSILON = 1e-2


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_open01(value: float) -> float:
    """Clamp score to strict open interval (0, 1)."""
    if not math.isfinite(value):
        return 0.5
    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, value))


def _has_boundary_terms(history: list[str]) -> bool:
    text = " ".join(history).lower()
    for term in ("scope", "paid", "revision", "contract", "milestone"):
        if term in text:
            return True
    return False


def grade_easy_task(result: EpisodeResult) -> float:
    """Simple deterministic grader for easy task."""
    task = TASKS["easy"]
    expected = task.expected_outcome
    ideal = float(expected["ideal_price"])
    max_steps = int(expected["max_steps_for_efficiency"])

    score = 0.2
    if result.decision == "accept":
        score += 0.4
    elif result.decision == "negotiate":
        score += 0.2

    if result.final_price is not None:
        if result.final_price >= ideal * 0.9:
            score += 0.3
        elif result.final_price >= ideal * 0.7:
            score += 0.15

    if 0 < result.step_count <= max_steps:
        score += 0.1

    return _clamp_open01(_clamp01(score))


def grade_medium_task(result: EpisodeResult) -> float:
    """Simple deterministic grader for medium task."""
    task = TASKS["medium"]
    expected = task.expected_outcome
    ideal = float(expected["ideal_price"])
    max_steps = int(expected["max_steps_for_efficiency"])

    score = 0.15
    if result.decision == "negotiate":
        score += 0.35
    elif result.decision == "accept":
        score += 0.25
    else:
        score += 0.1

    if result.final_price is not None:
        if result.final_price >= ideal * 0.9:
            score += 0.25
        elif result.final_price >= ideal * 0.75:
            score += 0.15
    else:
        score += 0.05

    if 2 <= result.step_count <= max_steps:
        score += 0.2
    elif result.step_count > 0:
        score += 0.1

    if len(result.conversation_history) >= 2:
        score += 0.1

    return _clamp_open01(_clamp01(score))


def grade_hard_task(result: EpisodeResult) -> float:
    """Simple deterministic grader for hard task."""
    task = TASKS["hard"]
    expected = task.expected_outcome
    min_viable = float(expected["min_viable_price"])
    max_steps = int(expected["max_steps_for_efficiency"])

    score = 0.1
    if result.decision == "reject":
        score += 0.45
    elif result.decision == "negotiate":
        score += 0.2

    if result.final_price is None and result.decision == "reject":
        score += 0.25
    elif result.final_price is not None and result.final_price >= min_viable:
        score += 0.25
    else:
        score += 0.05

    if 0 < result.step_count <= max_steps:
        score += 0.1

    if _has_boundary_terms(result.conversation_history):
        score += 0.1

    return _clamp_open01(_clamp01(score))


def grade_task(task_id: str, result: EpisodeResult) -> float:
    """Grade a task deterministically and return a score in strict range (0.0, 1.0)."""
    normalized_id = task_id.strip().lower()
    if normalized_id == "easy":
        return _clamp_open01(grade_easy_task(result))
    if normalized_id == "medium":
        return _clamp_open01(grade_medium_task(result))
    if normalized_id == "hard":
        return _clamp_open01(grade_hard_task(result))
    raise ValueError(f"Unknown task_id: {task_id}")


def get_tasks() -> list[TaskDefinition]:
    """Return all benchmark tasks in deterministic order."""
    return [TASKS["easy"], TASKS["medium"], TASKS["hard"]]
