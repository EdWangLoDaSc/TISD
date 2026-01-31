import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from agent_distill.utils.logging import logger


def compute_forward_kl(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(pi_T || pi_S) = sum_a pi_T(a) * [log pi_T(a) - log pi_S(a)].

    Args:
        teacher_logprobs: Log-probabilities from teacher, shape [num_actions].
        student_logprobs: Log-probabilities from student, shape [num_actions].

    Returns:
        Scalar KL divergence (non-negative).
    """
    teacher_probs = teacher_logprobs.exp()
    kl = (teacher_probs * (teacher_logprobs - student_logprobs)).sum()
    return kl


def apply_teacher_temperature(
    teacher_logprobs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Sharpen/soften teacher distribution with temperature.

    Args:
        teacher_logprobs: Log-probabilities, shape [num_actions].
        temperature: < 1.0 sharpens, > 1.0 softens.

    Returns:
        Temperature-adjusted log-probabilities (re-normalized).
    """
    if temperature == 1.0:
        return teacher_logprobs
    scaled_logits = teacher_logprobs / temperature
    return F.log_softmax(scaled_logits, dim=-1)


def compute_adaptive_weights(
    step_kls: torch.Tensor,
    lambda_: float = 2.0,
) -> torch.Tensor:
    """Compute softmax-normalized temporal weights from per-step KL values.

    w_t = exp(lambda * delta_t) / sum_j exp(lambda * delta_j)

    High-KL steps (where teacher and student disagree most) get higher weight,
    serving as adaptive credit assignment.

    Args:
        step_kls: KL divergences per step, shape [T]. Must be detached.
        lambda_: Temperature controlling weight concentration.

    Returns:
        Weights summing to 1, shape [T].
    """
    return F.softmax(lambda_ * step_kls, dim=0)


def tisd_trajectory_loss(
    teacher_logprobs_seq: List[torch.Tensor],
    student_logprobs_seq: List[torch.Tensor],
    lambda_: float = 2.0,
    kl_clip: float = 20.0,
    teacher_temp: float = 0.8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute TISD loss for a single trajectory.

    Steps:
        1. Apply teacher temperature sharpening to each step.
        2. Compute per-step forward KL: delta_t = KL(pi_T^sharp || pi_S).
        3. Clip KL: delta_t = min(delta_t, kl_clip).
        4. Compute adaptive weights: w_t = softmax(lambda * delta_t.detach()).
        5. Weighted loss: L = sum_t w_t * delta_t.

    Args:
        teacher_logprobs_seq: List of T tensors, each [num_actions_t].
        student_logprobs_seq: List of T tensors, each [num_actions_t].
        lambda_: Weight temperature.
        kl_clip: Maximum KL per step.
        teacher_temp: Teacher distribution temperature.

    Returns:
        loss: Scalar weighted KL loss.
        metrics: Dict with diagnostic info.
    """
    T = len(teacher_logprobs_seq)
    assert T == len(student_logprobs_seq), "Teacher and student must have same number of steps"

    if T == 0:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, {"loss": 0.0, "num_steps": 0}

    step_kls = []
    for t in range(T):
        # Apply temperature sharpening to teacher
        teacher_lp = apply_teacher_temperature(teacher_logprobs_seq[t], teacher_temp)
        student_lp = student_logprobs_seq[t]

        # Normalize student logprobs (they should already be log-probs over actions)
        student_lp = F.log_softmax(student_lp, dim=-1)

        kl_t = compute_forward_kl(teacher_lp, student_lp)
        # Clip to prevent extreme values
        kl_t = torch.clamp(kl_t, max=kl_clip)
        step_kls.append(kl_t)

    step_kls_tensor = torch.stack(step_kls)  # [T]

    # Adaptive weights (detached — no gradient through weighting)
    weights = compute_adaptive_weights(step_kls_tensor.detach(), lambda_)

    # Weighted loss
    loss = (weights * step_kls_tensor).sum()

    # Diagnostics
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "avg_kl": step_kls_tensor.mean().item(),
            "max_kl": step_kls_tensor.max().item(),
            "min_kl": step_kls_tensor.min().item(),
            "weight_entropy": -(weights * weights.log().clamp(min=-100)).sum().item(),
            "max_weight": weights.max().item(),
            "num_steps": T,
        }

    return loss, metrics


def tisd_batch_loss(
    batch_teacher_logprobs: List[List[torch.Tensor]],
    batch_student_logprobs: List[List[torch.Tensor]],
    lambda_: float = 2.0,
    kl_clip: float = 20.0,
    teacher_temp: float = 0.8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Average TISD loss across a batch of trajectories.

    Args:
        batch_teacher_logprobs: List of B trajectory logprob sequences.
        batch_student_logprobs: List of B trajectory logprob sequences.

    Returns:
        loss: Scalar average loss.
        metrics: Aggregated metrics.
    """
    B = len(batch_teacher_logprobs)
    assert B == len(batch_student_logprobs)

    if B == 0:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, {"loss": 0.0, "batch_size": 0}

    total_loss = torch.tensor(0.0, device=batch_student_logprobs[0][0].device)
    all_metrics = []

    for i in range(B):
        traj_loss, traj_metrics = tisd_trajectory_loss(
            batch_teacher_logprobs[i],
            batch_student_logprobs[i],
            lambda_=lambda_,
            kl_clip=kl_clip,
            teacher_temp=teacher_temp,
        )
        total_loss = total_loss + traj_loss
        all_metrics.append(traj_metrics)

    avg_loss = total_loss / B

    # Aggregate metrics
    agg = {
        "loss": avg_loss.item(),
        "batch_size": B,
        "avg_kl": sum(m["avg_kl"] for m in all_metrics) / B,
        "max_kl": max(m["max_kl"] for m in all_metrics),
        "avg_weight_entropy": sum(m["weight_entropy"] for m in all_metrics) / B,
        "avg_steps": sum(m["num_steps"] for m in all_metrics) / B,
    }

    return avg_loss, agg
