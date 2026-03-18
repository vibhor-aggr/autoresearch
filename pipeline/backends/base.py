"""
base.py – Abstract backend interface.

All compute backends must implement :meth:`run_experiment`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class AbstractBackend(ABC):
    """
    Abstract base class for all compute backends.

    A backend is responsible for:
    1. Receiving the path to a Python experiment script.
    2. Executing the script with the given arguments.
    3. Ensuring results are written to the path specified by ``--output``.
    """

    @abstractmethod
    def run_experiment(
        self,
        experiment_script: str,
        working_dir: str,
        extra_args: Dict[str, str],
    ) -> None:
        """
        Execute *experiment_script* on the backend.

        Parameters
        ----------
        experiment_script:
            Absolute or relative path to the Python file to run.
        working_dir:
            Directory to use as the working directory during execution.
        extra_args:
            Extra CLI arguments passed to the script as a dict, e.g.
            ``{"--seed": "0", "--output": "/path/to/results.json"}``.
        """
        ...
