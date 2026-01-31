from typing import List, Dict, Optional
from agent_distill.env.trajectory import Step


def build_student_prompt(
    instruction: str,
    icl_examples: list,
    task_description: str,
    steps: List[Step],
    t: int,
) -> List[Dict[str, str]]:
    """Build ChatML messages for the student at step t.

    The student sees only causal context: instruction + ICL examples +
    task description + history up to (but not including) step t's action.

    This reconstructs exactly what the agent would see at decision time.

    Args:
        instruction: ALFWorld instruction text.
        icl_examples: Raw ICL examples (list of conversation lists).
        task_description: Initial task observation.
        steps: Full trajectory steps.
        t: Current step index (we build prompt for deciding action at step t).

    Returns:
        messages: List of ChatML message dicts with "role" and "content" keys.
    """
    messages = []

    # System instruction
    messages.append({"role": "system", "content": instruction})

    # ICL examples as conversation turns
    for i, example in enumerate(icl_examples):
        for j, turn in enumerate(example):
            role = turn.get("role", "assistant" if turn.get("from") == "gpt" else "user")
            content = turn.get("content", turn.get("value", ""))
            if i == 0 and j == 0:
                # First ICL example starts with assistant acknowledging
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": "OK"})
            elif j == 0:
                # Start of subsequent ICL examples
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt") or turn.get("from") == "gpt":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})

    # Current task observation
    messages.append({"role": "user", "content": task_description})

    # Replay history up to step t
    for i in range(t):
        step = steps[i]
        # Agent's response
        messages.append({"role": "assistant", "content": step.llm_output})
        # Environment observation (for the NEXT step, which is step i+1's observation,
        # but we store it as the current step's observation feedback)
        if i < len(steps) - 1:
            messages.append({"role": "user", "content": f"Observation: {steps[i + 1].observation}"})
        elif i == len(steps) - 1 and i < t:
            # Last step before t, include its observation if available
            pass

    # For step t, the agent has received the observation (steps[t].observation)
    # and now needs to decide the action
    if t > 0:
        messages.append({"role": "user", "content": f"Observation: {steps[t].observation}"})
    # If t == 0, the task_description already serves as the observation

    return messages
