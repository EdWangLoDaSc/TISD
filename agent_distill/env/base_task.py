"""Abstract base class for tasks.

Ported from ETO/eval_agent/tasks/base.py with imports updated.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any


logger = logging.getLogger("tisd")


class Task(ABC):
    """Base class for a task instance."""

    task_name: str = "base"

    def __init__(self, **kwargs) -> None:
        self.task_id: Any = kwargs.get("task_id", None)

    @classmethod
    @abstractmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int) -> Tuple[List["Task"], int]:
        pass
