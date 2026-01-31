"""ALFWorld task loader.

Ported from ETO/eval_agent/tasks/alfworld.py with imports updated.
"""
import os
import yaml
import logging
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment as envs

from agent_distill.env.base_task import Task


logger = logging.getLogger("tisd")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """ALFWorld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        env: envs.AlfredTWEnv,
        obs: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs
        self.env = env

    @classmethod
    def load_tasks(
        cls,
        split: str,
        part_num: int,
        part_idx: int = -1,
        data_path: str = None,
    ) -> Tuple[Iterable[Task], int]:
        """Load ALFWorld tasks.

        Args:
            split: One of 'train', 'dev', 'test'.
            part_num: Number of partitions for parallel collection.
            part_idx: Which partition to use (-1 for all).
            data_path: Path to alfworld data directory. If None, uses
                       ALFWORLD_DATA env var.
        """
        if data_path is not None:
            os.environ["ALFWORLD_DATA"] = data_path
        alfworld_data_path = os.environ.get("ALFWORLD_DATA")

        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)

        if split == 'train':
            split = "train"
            N_TASKS = 3321
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134

        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=1)

        if part_num > 1:
            assert part_idx != -1
            part_inst_num = [N_TASKS // part_num] * part_num
            part_inst_num[-1] += N_TASKS % part_num
            env.skip(sum(part_inst_num[:part_idx]))
            N_TASKS = part_inst_num[part_idx]

        def generator():
            for idx in range(N_TASKS):
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                yield cls(
                    task_id=idx,
                    game_file=game_file,
                    env=env,
                    obs=obs,
                )

        return generator(), N_TASKS
