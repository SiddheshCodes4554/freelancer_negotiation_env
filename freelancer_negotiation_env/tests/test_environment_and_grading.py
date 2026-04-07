from freelancer_negotiation_env.models import FreelancerNegotiationAction, NegotiationActionType
from freelancer_negotiation_env.server.freelancer_negotiation_env_environment import (
    FreelancerNegotiationEnvironment,
)
from freelancer_negotiation_env.tasks import (
    EpisodeResult,
    TASKS,
    get_tasks,
    grade_easy_task,
    grade_hard_task,
    grade_medium_task,
    grade_task,
)


def test_extract_price_from_text_inr_patterns() -> None:
    parser = FreelancerNegotiationEnvironment._extract_price_from_text

    assert parser("I can do this for Rs 3,000") == 3000.0
    assert parser("Final quote: INR 12,500.50") == 12500.50
    assert parser("Let us close at ₹4500") == 4500.0
    assert parser("No numeric offer in this message") is None


def test_accept_action_closes_episode() -> None:
    env = FreelancerNegotiationEnvironment()
    env.reset()

    action = FreelancerNegotiationAction(
        message="I accept and we can proceed at Rs 1300.",
        action_type=NegotiationActionType.ACCEPT,
    )
    obs = env.step(action)

    assert obs.done is True
    assert "Confirmed" in obs.client_message


def test_graders_are_bounded_for_all_tasks() -> None:
    sample = EpisodeResult(
        final_price=1250.0,
        decision="negotiate",
        conversation_history=[
            "freelancer: Let us align scope, budget, timeline, and revision terms.",
            "client: Please share a proposal.",
        ],
        step_count=3,
        client_type="normal",
    )

    for task in get_tasks():
        score = grade_task(task.task_id, sample)
        assert 0.0 < score < 1.0


def test_direct_task_graders_are_strictly_open_interval() -> None:
    low_quality = EpisodeResult(
        final_price=None,
        decision="accept",
        conversation_history=[],
        step_count=0,
        client_type="normal",
    )
    high_quality = EpisodeResult(
        final_price=float(TASKS["easy"].expected_outcome["ideal_price"]),
        decision="accept",
        conversation_history=[
            "freelancer: Scope, budget, timeline, and contract are aligned at Rs 1250.",
            "client: Great, proceeding now.",
        ],
        step_count=1,
        client_type="normal",
    )

    for grader in (grade_easy_task, grade_medium_task, grade_hard_task):
        lo = grader(low_quality)
        hi = grader(high_quality)
        assert 0.0 < lo < 1.0
        assert 0.0 < hi < 1.0


def test_memory_summary_keeps_recent_three_deals() -> None:
    env = FreelancerNegotiationEnvironment()

    for idx in range(5):
        env.reset()
        action = FreelancerNegotiationAction(
            message=f"I accept at Rs {1200 + idx * 25}.",
            action_type=NegotiationActionType.ACCEPT,
        )
        env.step(action)

    assert len(env.memory_summary) == 3
    assert all("client_type" in item for item in env.memory_summary)
