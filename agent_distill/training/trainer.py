import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

from agent_distill.env.trajectory import Trajectory
from agent_distill.model.qwen_wrapper import QwenModel
from agent_distill.prompts.student import build_student_prompt
from agent_distill.prompts.teacher import build_teacher_prompt
from agent_distill.distill.kl_loss import tisd_trajectory_loss
from agent_distill.utils.logging import logger, log_metrics


class TISDTrainer:
    """Handles one iteration of TISD distillation training."""

    def __init__(self, model: QwenModel, config: dict, env_config: dict):
        self.model = model
        self.config = config
        self.tisd_config = config.get("tisd", {})
        self.train_config = config.get("training", {})

        # Load instruction and ICL for prompt building
        with open(env_config["instruction_path"]) as f:
            self.instruction = f.read()
        with open(env_config["icl_path"]) as f:
            self.icl_examples = json.load(f)

        # TISD hyperparameters
        self.lambda_ = self.tisd_config.get("lambda_", 2.0)
        self.teacher_temp = self.tisd_config.get("teacher_temp", 0.8)
        self.kl_clip = self.tisd_config.get("kl_clip", 20.0)

        # Training hyperparameters
        self.lr = self.train_config.get("lr", 2e-5)
        self.grad_accum_steps = self.train_config.get("grad_accumulation_steps", 4)
        self.max_grad_norm = self.train_config.get("max_grad_norm", 1.0)
        self.batch_size = self.train_config.get("batch_size", 4)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.lr,
            weight_decay=self.train_config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        self.scheduler = None  # Set up during training with known total steps

    def _get_action_space(self, trajectory: Trajectory, t: int) -> List[str]:
        """Get the action space for step t.

        Uses admissible_actions if available, otherwise constructs a minimal set
        from the actual action and common ALFWorld actions.
        """
        step = trajectory.steps[t]
        if step.admissible_actions:
            return step.admissible_actions

        # Fallback: use the actual action taken plus generic ALFWorld actions
        # This is a simplification — ideally we'd record admissible_commands during collection
        actions = [step.action]

        # Add actions from nearby steps for diversity
        for i in range(max(0, t - 2), min(len(trajectory.steps), t + 3)):
            if trajectory.steps[i].action and trajectory.steps[i].action not in actions:
                actions.append(trajectory.steps[i].action)

        return actions

    def distill_step(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """One gradient update from a batch of trajectories.

        For each trajectory, for each step:
          1. Build student prompt -> get student action logprobs (with grad)
          2. Build teacher prompt -> get teacher action logprobs (no grad)
          3. Compute per-step KL, adaptive weights, weighted loss

        Args:
            trajectories: Batch of trajectories.

        Returns:
            metrics: Dict with loss and diagnostic info.
        """
        self.model.model.train()
        total_loss = torch.tensor(0.0, device=self.model.device)
        all_traj_metrics = []

        for traj_idx, traj in enumerate(trajectories):
            if len(traj) == 0:
                continue

            teacher_logprobs_seq = []
            student_logprobs_seq = []

            for t in range(len(traj)):
                actions = self._get_action_space(traj, t)
                if len(actions) < 2:
                    # Need at least 2 actions for meaningful KL
                    continue

                # Student prompt: causal context only
                student_msgs = build_student_prompt(
                    instruction=self.instruction,
                    icl_examples=self.icl_examples[:self.config.get("env", {}).get("icl_num", 1)],
                    task_description=traj.task_description,
                    steps=traj.steps,
                    t=t,
                )

                # Teacher prompt: with hindsight
                teacher_msgs = build_teacher_prompt(
                    instruction=self.instruction,
                    icl_examples=self.icl_examples[:self.config.get("env", {}).get("icl_num", 1)],
                    task_description=traj.task_description,
                    trajectory=traj,
                    t=t,
                )

                # Get logprobs: student WITH grad, teacher WITHOUT grad
                student_lp = self.model.get_action_logprobs(
                    student_msgs, actions, require_grad=True
                )
                teacher_lp = self.model.get_action_logprobs(
                    teacher_msgs, actions, require_grad=False
                )

                # Normalize to proper log-distributions
                student_logprobs_seq.append(F.log_softmax(student_lp, dim=-1))
                teacher_logprobs_seq.append(F.log_softmax(teacher_lp, dim=-1))

            if len(student_logprobs_seq) == 0:
                continue

            # Compute TISD loss for this trajectory
            traj_loss, traj_metrics = tisd_trajectory_loss(
                teacher_logprobs_seq=teacher_logprobs_seq,
                student_logprobs_seq=student_logprobs_seq,
                lambda_=self.lambda_,
                kl_clip=self.kl_clip,
                teacher_temp=self.teacher_temp,
            )

            total_loss = total_loss + traj_loss / len(trajectories)
            all_traj_metrics.append(traj_metrics)

        # Gradient step (with accumulation support)
        if total_loss.requires_grad:
            scaled_loss = total_loss / self.grad_accum_steps
            scaled_loss.backward()

        # Aggregate metrics
        if all_traj_metrics:
            metrics = {
                "loss": total_loss.item(),
                "avg_kl": sum(m["avg_kl"] for m in all_traj_metrics) / len(all_traj_metrics),
                "max_kl": max(m["max_kl"] for m in all_traj_metrics),
                "avg_weight_entropy": sum(m["weight_entropy"] for m in all_traj_metrics) / len(all_traj_metrics),
                "avg_steps_per_traj": sum(m["num_steps"] for m in all_traj_metrics) / len(all_traj_metrics),
                "num_trajectories": len(all_traj_metrics),
            }
        else:
            metrics = {"loss": 0.0, "num_trajectories": 0}

        return metrics

    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        torch.nn.utils.clip_grad_norm_(
            self.model.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(
        self,
        trajectories: List[Trajectory],
        epoch: int,
        global_step: int,
    ) -> int:
        """Train one epoch over all collected trajectories.

        Args:
            trajectories: All trajectories for this iteration.
            epoch: Current epoch number.
            global_step: Current global step counter.

        Returns:
            Updated global_step.
        """
        self.optimizer.zero_grad()
        batch_size = self.batch_size
        accum_count = 0

        for batch_start in range(0, len(trajectories), batch_size):
            batch = trajectories[batch_start : batch_start + batch_size]
            metrics = self.distill_step(batch)
            accum_count += 1

            if accum_count >= self.grad_accum_steps:
                self.optimizer_step()
                accum_count = 0
                global_step += 1

                log_every = self.config.get("logging", {}).get("log_every", 10)
                if global_step % log_every == 0:
                    log_metrics(metrics, global_step, prefix=f"epoch_{epoch}")

        # Handle remaining accumulated gradients
        if accum_count > 0:
            self.optimizer_step()
            global_step += 1

        return global_step

    def save_checkpoint(self, path: str, iteration: int, global_step: int):
        """Save model checkpoint."""
        save_path = os.path.join(path, f"iter_{iteration}")
        os.makedirs(save_path, exist_ok=True)

        self.model.model.save_pretrained(save_path)
        self.model.tokenizer.save_pretrained(save_path)

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
            "iteration": iteration,
        }, os.path.join(save_path, "training_state.pt"))

        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint directory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading checkpoint from {path}")
        self.model.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=self.model.dtype, trust_remote_code=True
        ).to(self.model.device)
        self.model.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, padding_side="left"
        )

        # Load optimizer state if available
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.model.device)
            self.optimizer.load_state_dict(state["optimizer"])
            return state.get("global_step", 0), state.get("iteration", 0)
        return 0, 0
