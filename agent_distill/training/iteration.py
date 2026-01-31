import os
import yaml
import json
from typing import Optional

from agent_distill.model.qwen_wrapper import QwenModel
from agent_distill.env.alfworld_adapter import ALFWorldCollector
from agent_distill.training.trainer import TISDTrainer
from agent_distill.utils.logging import logger, setup_logging, setup_wandb


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class TISDPipeline:
    """Orchestrates the full iterative TISD training pipeline.

    Each iteration:
        1. Collect trajectories with current policy
        2. Run distillation training on collected trajectories
        3. Evaluate on dev/test set
        4. Save checkpoint
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model = QwenModel.from_config(self.config["model"])
        self.collector = ALFWorldCollector(self.config["env"])
        self.trainer = TISDTrainer(
            model=self.model,
            config=self.config,
            env_config=self.config["env"],
        )

        self.save_dir = self.config["logging"]["save_dir"]
        self.traj_dir = self.config["logging"]["trajectory_dir"]
        self.num_iterations = self.config["training"]["num_iterations"]
        self.distill_epochs = self.config["training"]["distill_epochs"]
        self.num_collect = self.config["env"]["num_collect"]
        self.num_eval = self.config["env"].get("num_eval", 134)

    def run_iteration(self, iteration: int, global_step: int) -> int:
        """Execute one full TISD iteration.

        Args:
            iteration: Iteration number (0-indexed).
            global_step: Current global training step.

        Returns:
            Updated global_step.
        """
        logger.info(f"=== TISD Iteration {iteration + 1}/{self.num_iterations} ===")

        # Phase 1: Collect trajectories
        logger.info("Phase 1: Collecting trajectories...")
        traj_save_dir = os.path.join(self.traj_dir, f"iter_{iteration}")
        os.makedirs(traj_save_dir, exist_ok=True)

        self.model.model.eval()
        trajectories = self.collector.collect_batch(
            model=self.model,
            num_trajectories=self.num_collect,
            split="train",
        )

        # Save trajectories
        for i, traj in enumerate(trajectories):
            traj.save(os.path.join(traj_save_dir, f"traj_{i:04d}.json"))

        # Log collection stats
        successes = sum(1 for t in trajectories if t.success)
        logger.info(f"Collected {len(trajectories)} trajectories, "
                    f"{successes} successes ({successes / len(trajectories) * 100:.1f}%)")

        # Phase 2: Distillation training
        logger.info("Phase 2: Distillation training...")
        for epoch in range(self.distill_epochs):
            logger.info(f"  Distill epoch {epoch + 1}/{self.distill_epochs}")
            global_step = self.trainer.train_epoch(
                trajectories=trajectories,
                epoch=epoch,
                global_step=global_step,
            )

        # Phase 3: Evaluation
        logger.info("Phase 3: Evaluating...")
        self.model.model.eval()
        eval_metrics = self.collector.evaluate(
            model=self.model,
            split="test",
            num_tasks=self.num_eval,
        )
        logger.info(f"Evaluation results: {eval_metrics}")

        # Save eval metrics
        metrics_path = os.path.join(self.save_dir, f"iter_{iteration}", "eval_metrics.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        # Phase 4: Save checkpoint
        self.trainer.save_checkpoint(self.save_dir, iteration, global_step)

        return global_step

    def run(
        self,
        num_iterations: Optional[int] = None,
        resume_iteration: int = 0,
    ):
        """Full TISD training pipeline.

        Args:
            num_iterations: Override for number of iterations.
            resume_iteration: Start from this iteration (for resuming).
        """
        n_iter = num_iterations or self.num_iterations
        log_dir = setup_logging(self.config)
        setup_wandb(self.config)

        logger.info("Starting TISD training pipeline")
        logger.info(f"Model: {self.config['model']['name_or_path']}")
        logger.info(f"Iterations: {n_iter}")
        logger.info(f"Trajectories per iteration: {self.num_collect}")

        # Initial evaluation before any training
        if resume_iteration == 0:
            logger.info("=== Initial Evaluation (before training) ===")
            self.model.model.eval()
            init_metrics = self.collector.evaluate(
                model=self.model,
                split="test",
                num_tasks=min(self.num_eval, 50),  # Quick initial eval
            )
            logger.info(f"Initial metrics: {init_metrics}")

        global_step = 0
        for iteration in range(resume_iteration, n_iter):
            global_step = self.run_iteration(iteration, global_step)

        logger.info("TISD training complete!")
