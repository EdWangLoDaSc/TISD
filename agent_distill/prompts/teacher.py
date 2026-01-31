from typing import List, Dict
from agent_distill.env.trajectory import Step, Trajectory


HINDSIGHT_TEMPLATE = """{instruction}

[RETROSPECTIVE EVALUATION MODE]
You are reviewing step {step_num} of a completed trajectory ({total_steps} total steps).

After this step, the following actions and observations occurred:
{future_trajectory}

Final outcome: {outcome}

Given this complete hindsight, consider what the best action would have been at the current step."""


def _format_future(steps: List[Step], t: int) -> str:
    """Format the future trajectory after step t."""
    lines = []
    for i in range(t + 1, len(steps)):
        s = steps[i]
        lines.append(f"Step {i + 1}: Action: {s.action}")
        lines.append(f"  Observation: {s.observation}")
    if not lines:
        return "(This was the final step)"
    return "\n".join(lines)


def build_teacher_prompt(
    instruction: str,
    icl_examples: list,
    task_description: str,
    trajectory: Trajectory,
    t: int,
) -> List[Dict[str, str]]:
    """Build ChatML messages for the teacher at step t.

    The teacher sees everything the student sees PLUS hindsight:
    - The full future trajectory after step t
    - The final outcome (success/failure)

    Hindsight is injected into the system message so that it contextualizes
    all subsequent reasoning without altering the conversation structure.

    Args:
        instruction: ALFWorld instruction text.
        icl_examples: Raw ICL examples.
        task_description: Initial task observation.
        trajectory: Complete trajectory with outcome.
        t: Step index being evaluated.

    Returns:
        messages: List of ChatML message dicts.
    """
    steps = trajectory.steps

    # Build hindsight-augmented system message
    future_text = _format_future(steps, t)
    outcome = "SUCCESS" if trajectory.success else "FAILURE"
    system_content = HINDSIGHT_TEMPLATE.format(
        instruction=instruction,
        step_num=t + 1,
        total_steps=len(steps),
        future_trajectory=future_text,
        outcome=outcome,
    )

    messages = [{"role": "system", "content": system_content}]

    # ICL examples (same as student)
    for i, example in enumerate(icl_examples):
        for j, turn in enumerate(example):
            role = turn.get("role", "assistant" if turn.get("from") == "gpt" else "user")
            content = turn.get("content", turn.get("value", ""))
            if i == 0 and j == 0:
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": "OK"})
            elif j == 0:
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt") or turn.get("from") == "gpt":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})

    # Current task observation
    messages.append({"role": "user", "content": task_description})

    # Replay history up to step t (same as student)
    for i in range(t):
        step = steps[i]
        messages.append({"role": "assistant", "content": step.llm_output})
        if i < len(steps) - 1:
            messages.append({"role": "user", "content": f"Observation: {steps[i + 1].observation}"})

    if t > 0:
        messages.append({"role": "user", "content": f"Observation: {steps[t].observation}"})

    return messages
