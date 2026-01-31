import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Step:
    """A single step in an agent trajectory."""
    observation: str            # Environment observation at this step
    action: str                 # Parsed action (e.g. "go to shelf 1")
    thought: str                # Agent's reasoning before acting
    llm_output: str             # Raw LLM output ("Thought: ...\nAction: ...")
    admissible_actions: List[str] = field(default_factory=list)  # Valid actions at this state

    @staticmethod
    def parse_llm_output(llm_output: str) -> tuple:
        """Parse raw LLM output into (thought, action)."""
        thought, action = "", ""
        llm_output = llm_output.strip()
        if "Action:" in llm_output:
            parts = llm_output.split("Action:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            action = parts[1].strip()
        else:
            thought = llm_output.replace("Thought:", "").strip()
        return thought, action


@dataclass
class Trajectory:
    """A complete episode trajectory."""
    task_id: str
    task_description: str       # Initial observation / task prompt
    game_file: str              # ALFWorld game file path
    steps: List[Step]
    success: bool
    total_reward: float         # 0.0 or 1.0 for ALFWorld

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "game_file": self.game_file,
            "steps": [asdict(s) for s in self.steps],
            "success": self.success,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        steps = [Step(**s) for s in d["steps"]]
        return cls(
            task_id=d["task_id"],
            task_description=d["task_description"],
            game_file=d["game_file"],
            steps=steps,
            success=d["success"],
            total_reward=d["total_reward"],
        )

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_batch(cls, directory: str) -> List["Trajectory"]:
        trajectories = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".json"):
                trajectories.append(cls.load(os.path.join(directory, fname)))
        return trajectories

    def __len__(self):
        return len(self.steps)
