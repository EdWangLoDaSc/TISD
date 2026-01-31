"""Main entry point for TISD training.

Usage:
    python -m agent_distill.scripts.run_tisd
    python -m agent_distill.scripts.run_tisd --config /path/to/custom.yaml
    python -m agent_distill.scripts.run_tisd --resume_iter 1
"""
import argparse

from agent_distill.training.iteration import TISDPipeline


def main():
    parser = argparse.ArgumentParser(description="TISD: Trajectory-Informed Self-Distillation")
    parser.add_argument(
        "--config", type=str, default="default.yaml",
        help="Config file name (looked up in agent_distill/configs/) or absolute path.",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=None,
        help="Override number of TISD iterations.",
    )
    parser.add_argument(
        "--resume_iter", type=int, default=0,
        help="Resume from this iteration number.",
    )
    args = parser.parse_args()

    pipeline = TISDPipeline(args.config)
    pipeline.run(
        num_iterations=args.num_iterations,
        resume_iteration=args.resume_iter,
    )


if __name__ == "__main__":
    main()
