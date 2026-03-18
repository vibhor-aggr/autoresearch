"""
local.py – Local backend: run experiments in a subprocess.

Suitable for:
  - Agent mode (GitHub Copilot / Codespaces)
  - Any machine where the pipeline runs directly

The experiment script is executed as:
    python <experiment_script> [extra_args...]
"""

from __future__ import annotations

import subprocess
import sys
from typing import Dict

from pipeline.backends.base import AbstractBackend
from pipeline.config import BackendConfig


class LocalBackend(AbstractBackend):
    """
    Executes experiment scripts in a local subprocess.

    Uses the same Python interpreter that is currently running the pipeline
    (``sys.executable``), so experiment dependencies must be installed in the
    same environment.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    def run_experiment(
        self,
        experiment_script: str,
        working_dir: str,
        extra_args: Dict[str, str],
    ) -> None:
        """
        Run ``python <experiment_script> <extra_args>`` in *working_dir*.

        Raises
        ------
        subprocess.CalledProcessError
            If the script exits with a non-zero status.
        """
        cmd = [sys.executable, experiment_script]
        for key, value in extra_args.items():
            cmd.extend([key, value])

        subprocess.run(
            cmd,
            cwd=working_dir,
            check=True,
            capture_output=False,
        )
