"""Standalone evaluation script.

Usage:
    python -m agent_distill.scripts.evaluate \
        --config agent_distill/configs/default.yaml \
        --split test --num 134
    python -m agent_distill.scripts.evaluate \
        --config agent_distill/configs/default.yaml \
        --model_path checkpoints/iter_2 --split test
"""
import argparse
import json
import yaml

from agent_distill.model.qwen_wrapper import QwenModel
from agent_distill.env.alfworld_adapter import ALFWorldCollector
from agent_distill.utils.logging import logger, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on ALFWorld")
    parser.add_argument("--config", type=str, default="agent_distill/configs/default.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num", type=int, default=None, help="Number of tasks (None=all)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (e.g. a checkpoint)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save eval results JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    model_path = args.model_path or config["model"]["name_or_path"]
    config["model"]["name_or_path"] = model_path

    model = QwenModel.from_config(config["model"])
    collector = ALFWorldCollector(config["env"])

    logger.info(f"Evaluating {model_path} on {args.split} split")
    metrics = collector.evaluate(
        model=model,
        split=args.split,
        num_tasks=args.num,
    )

    logger.info(f"Results: {json.dumps(metrics, indent=2)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
