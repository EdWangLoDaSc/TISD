import os
import json
from typing import List, Optional

from agent_distill.env.trajectory import Trajectory, Step
from agent_distill.env.alfworld_task import AlfWorldTask
from agent_distill.env.alfworld_env import AlfWorldEnv
from agent_distill.utils.logging import logger


class ALFWorldCollector:
    """Collects trajectories by running a model in ALFWorld."""

    def __init__(self, config: dict):
        self.config = config
        self.max_steps = config.get("max_steps", 40)
        self.split = config.get("split", "train")
        self.data_path = config.get("data_path", "agent_distill/data/alfworld")

        # Load instruction and ICL examples
        with open(config["instruction_path"]) as f:
            self.instruction = f.read()
        with open(config["icl_path"]) as f:
            self.icl_examples = json.load(f)
        self.icl_num = config.get("icl_num", 1)

    def _init_env(self, split: str, part_num: int = 1, part_idx: int = -1):
        """Initialize ALFWorld task generator."""
        return AlfWorldTask.load_tasks(
            split, part_num, part_idx, data_path=self.data_path
        )

    def collect_trajectory(self, model, task_obj) -> Trajectory:
        """Run one episode and return a Trajectory.

        Args:
            model: QwenModel instance with generate() method.
            task_obj: An AlfWorldTask instance.

        Returns:
            Trajectory object with all steps and outcome.
        """
        env = AlfWorldEnv(
            task=task_obj,
            instruction_path=self.config["instruction_path"],
            icl_path=self.config["icl_path"],
            icl_format="conversation",
            max_steps=self.max_steps,
        )
        observation, state = env.reset()

        task_description = task_obj.observation
        game_file = task_obj.game_file
        task_id = str(task_obj.task_id)

        steps = []

        while not state.finished:
            # Build messages from state history for generation
            messages = self._history_to_messages(state.history)

            # Generate action
            gen_config = self.config.get("generation", {})
            llm_output = model.generate(
                messages,
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                max_new_tokens=gen_config.get("max_new_tokens", 128),
            )

            # Parse thought and action from LLM output
            thought, action = Step.parse_llm_output(llm_output)

            # Record step
            current_obs = self._extract_last_observation(state.history)
            step = Step(
                observation=current_obs,
                action=action,
                thought=thought,
                llm_output=llm_output,
                admissible_actions=[],
            )
            steps.append(step)

            # Execute in environment
            observation, state = env.step(llm_output)

        return Trajectory(
            task_id=task_id,
            task_description=task_description,
            game_file=game_file,
            steps=steps,
            success=state.success,
            total_reward=float(state.reward) if state.reward is not None else 0.0,
        )

    def collect_batch(
        self,
        model,
        num_trajectories: int,
        split: Optional[str] = None,
    ) -> List[Trajectory]:
        """Collect a batch of trajectories.

        Args:
            model: QwenModel instance.
            num_trajectories: Number of trajectories to collect.
            split: Data split ("train", "dev", "test"). Defaults to config split.

        Returns:
            List of Trajectory objects.
        """
        split = split or self.split
        task_gen, n_tasks = self._init_env(split)

        trajectories = []
        for i, task in enumerate(task_gen):
            if i >= num_trajectories:
                break

            logger.info(f"Collecting trajectory {i + 1}/{num_trajectories} "
                        f"(task_id={task.task_id})")
            try:
                traj = self.collect_trajectory(model, task)
                trajectories.append(traj)
                logger.info(f"  -> {'SUCCESS' if traj.success else 'FAILURE'} "
                            f"in {len(traj)} steps")
            except Exception as e:
                logger.warning(f"  -> Failed to collect trajectory: {e}")
                continue

        return trajectories

    def _history_to_messages(self, history: list) -> List[dict]:
        """Convert state.history to ChatML messages for Qwen."""
        messages = [{"role": "system", "content": self.instruction}]
        for entry in history:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        return messages

    @staticmethod
    def _extract_last_observation(history: list) -> str:
        """Get the most recent user message (observation) from history."""
        for entry in reversed(history):
            if entry["role"] == "user":
                content = entry["content"]
                if content.startswith("Observation: "):
                    return content[len("Observation: "):]
                return content
        return ""

    def evaluate(self, model, split: str = "test", num_tasks: Optional[int] = None) -> dict:
        """Evaluate model on a split and return metrics.

        Args:
            model: QwenModel instance.
            split: Evaluation split.
            num_tasks: Max tasks to evaluate. None = all tasks in split.

        Returns:
            Dict with success_rate, avg_steps, num_success, num_total.
        """
        task_gen, n_tasks = self._init_env(split)
        if num_tasks is not None:
            n_tasks = min(n_tasks, num_tasks)

        successes = 0
        total_steps = 0
        total = 0

        for i, task in enumerate(task_gen):
            if i >= n_tasks:
                break

            try:
                traj = self.collect_trajectory(model, task)
                if traj.success:
                    successes += 1
                total_steps += len(traj)
                total += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Eval [{i + 1}/{n_tasks}] "
                                f"SR={successes / total:.3f}")
            except Exception as e:
                logger.warning(f"Eval task {i} failed: {e}")
                total += 1

        metrics = {
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_steps": total_steps / total if total > 0 else 0.0,
            "num_success": successes,
            "num_total": total,
        }
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
