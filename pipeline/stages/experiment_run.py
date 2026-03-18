"""
experiment_run.py – Stage 4: Execute the experiment on the chosen backend.

Dispatches to one of:
  LocalBackend      – runs ``python run_experiment.py --output results.json``
                      in a subprocess on the current machine.
  KaggleBackend     – pushes code to a Kaggle Dataset, creates a Kernel, waits
                      for completion, downloads results.
  RemoteSSHBackend  – SSHes into a remote server, copies code, executes, and
                      fetches results via SFTP.

Output stored in ``state.stage_outputs["EXPERIMENT_RUN"]``:
  {
    "results_files": [str, ...],   # paths to result JSON files
    "raw_results":   [{...}, ...], # parsed JSON from each results file
    "backend_used":  str,
    "runtime_seconds": float,
  }
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import append_log_entry

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig
    from pipeline.state import IdeaState
    from pipeline.utils.llm import LLMClient


class ExperimentRunStage(BaseStage):
    stage_name = "EXPERIMENT_RUN"

    def __init__(self, config: "PipelineConfig", llm: "LLMClient") -> None:
        super().__init__(config, llm)
        self._backend = _build_backend(config)

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        exp_dir = os.path.abspath(self._experiments_dir(idea_dir))
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        hyp_out     = state.stage_outputs.get("HYPOTHESIS", {})
        plan_out    = state.stage_outputs.get("EXPERIMENT_PLAN", {})
        min_runs    = int(hyp_out.get("min_runs", 3))
        # Cap at 3 for demo/CI purposes; override in config if desired
        min_runs    = min(min_runs, 3)

        results_files: List[str] = []
        raw_results:   List[Dict[str, Any]] = []

        start = time.monotonic()
        for seed in range(min_runs):
            out_file = os.path.join(results_dir, f"run_seed{seed}.json")
            self._backend.run_experiment(
                experiment_script=os.path.join(exp_dir, "run_experiment.py"),
                working_dir=exp_dir,
                extra_args={"--seed": str(seed), "--output": out_file},
            )
            if os.path.exists(out_file):
                with open(out_file, encoding="utf-8") as fh:
                    raw_results.append(json.load(fh))
                results_files.append(out_file)

        elapsed = time.monotonic() - start

        output: Dict[str, Any] = {
            "results_files": results_files,
            "raw_results":   raw_results,
            "backend_used":  self.config.backend.type,
            "runtime_seconds": round(elapsed, 1),
        }

        append_log_entry(
            self._log_path(idea_dir),
            step=4,
            action=(
                f"Experiments executed ({min_runs} runs) "
                f"on backend '{self.config.backend.type}'"
            ),
            reasoning=(
                f"Runtime: {elapsed:.1f}s.  "
                f"Collected {len(results_files)} result file(s)."
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Backend builder
# ---------------------------------------------------------------------------

def _build_backend(config: "PipelineConfig"):
    btype = config.backend.type.lower()
    if btype == "kaggle":
        from pipeline.backends.kaggle_backend import KaggleBackend
        return KaggleBackend(config.backend)
    if btype == "remote":
        from pipeline.backends.remote_backend import RemoteSSHBackend
        return RemoteSSHBackend(config.backend)
    # Default: local
    from pipeline.backends.local import LocalBackend
    return LocalBackend(config.backend)
