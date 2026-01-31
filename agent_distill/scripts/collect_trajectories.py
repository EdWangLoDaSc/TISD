"""Standalone trajectory collection.

Usage:
    python -m agent_distill.scripts.collect_trajectories
    python -m agent_distill.scripts.collect_trajectories --num 64 --split train
"""
import argparse
import os

from agent_distill.training.iteration import load_config
from agent_distill.model.qwen_wrapper import QwenModel
from agent_distill.env.alfworld_adapter import ALFWorldCollector
from agent_distill.utils.logging import logger, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Collect ALFWorld trajectories")
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--num", type=int, default=64, help="Number of trajectories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: <trajectory_dir>/collected)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (e.g. a checkpoint)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    model_path = args.model_path or config["model"]["name_or_path"]
    config["model"]["name_or_path"] = model_path

    output = args.output or os.path.join(config["logging"]["trajectory_dir"], "collected")

    model = QwenModel.from_config(config["model"])
    collector = ALFWorldCollector(config["env"])

    logger.info(f"Collecting {args.num} trajectories from {args.split} split")
    trajectories = collector.collect_batch(
        model=model,
        num_trajectories=args.num,
        split=args.split,
    )

    os.makedirs(output, exist_ok=True)
    for i, traj in enumerate(trajectories):
        traj.save(os.path.join(output, f"traj_{i:04d}.json"))

    successes = sum(1 for t in trajectories if t.success)
    logger.info(f"Done. {len(trajectories)} trajectories saved to {output}")
    logger.info(f"Success rate: {successes}/{len(trajectories)} "
                f"({successes / len(trajectories) * 100:.1f}%)")


if __name__ == "__main__":
    main()
