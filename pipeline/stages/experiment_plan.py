"""
experiment_plan.py – Stage 3: Plan experiments and generate runnable code.

The LLM receives the idea + hypothesis and produces:
  - A self-contained Python experiment script (``run_experiment.py``)
  - A ``requirements.txt``
  - A step-by-step experimental plan written into the specification

Output stored in ``state.stage_outputs["EXPERIMENT_PLAN"]``:
  {
    "plan_summary": str,
    "datasets": [str, ...],
    "baselines": [str, ...],
    "hyperparameters": {str: list},
    "experiment_code": str,   # full Python source
    "requirements": [str, ...],
    "expected_runtime_hours": float,
    "compute_recommendation": str,  # "local" | "kaggle" | "remote"
  }
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import (
    append_log_entry,
    read_text,
    write_experiment_code,
    write_requirements,
    write_text,
)

if TYPE_CHECKING:
    from pipeline.state import IdeaState


_SYSTEM = """\
You are a skilled ML engineer who writes clean, self-contained Python experiment scripts.
Respond with valid JSON only. All code must be placed inside the JSON values as strings.
"""

_USER_TEMPLATE = """\
Idea:
  Title       : {title}
  Description : {description}

Hypothesis:
  H₀          : {h0}
  H₁          : {h1}
  Metrics     : {metrics}
  Acceptance  : {acceptance_criteria}

Generate a complete experimental plan and a self-contained Python script.

Return a JSON object with exactly these keys:
  "plan_summary"            : 3–5 sentence description of the experimental design
  "datasets"                : list of dataset names
  "baselines"               : list of baseline method names
  "hyperparameters"         : dict mapping param name → list of values to try
  "experiment_code"         : complete Python source for run_experiment.py
                              (must save results to a JSON file given by --output arg;
                               must be runnable with: python run_experiment.py --output results.json)
  "requirements"            : list of pip package specs (e.g. ["torch>=2.0", "numpy"])
  "expected_runtime_hours"  : estimated wall-clock hours on a single GPU
  "compute_recommendation"  : "local" | "kaggle" | "remote"
"""

_FALLBACK_CODE = '''\
"""run_experiment.py – Stub experiment (auto-generated)."""
import argparse, json, random, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    # Simulate a short experiment
    time.sleep(1)
    results = {
        "method": "proposed",
        "baseline_metric": round(random.uniform(0.20, 0.30), 4),
        "proposed_metric": round(random.uniform(0.10, 0.20), 4),
        "accuracy_baseline": round(random.uniform(90.0, 92.0), 2),
        "accuracy_proposed": round(random.uniform(90.0, 92.0), 2),
        "seed": 42,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
'''


class ExperimentPlanStage(BaseStage):
    stage_name = "EXPERIMENT_PLAN"

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        idea_out = state.stage_outputs.get("IDEA_GENERATION", {})
        hyp_out  = state.stage_outputs.get("HYPOTHESIS", {})

        user_msg = _USER_TEMPLATE.format(
            title=idea_out.get("title", ""),
            description=idea_out.get("description", ""),
            h0=hyp_out.get("h0", ""),
            h1=hyp_out.get("h1", ""),
            metrics=hyp_out.get("metrics", []),
            acceptance_criteria=hyp_out.get("acceptance_criteria", {}),
        )

        output = self.llm.complete_json(_SYSTEM, user_msg)
        output = _normalise(output)

        exp_dir = self._experiments_dir(idea_dir)

        # Write generated experiment code
        write_experiment_code(exp_dir, "run_experiment.py", output["experiment_code"])
        write_requirements(exp_dir, output["requirements"])

        # Append experiment plan to specification.md
        _append_plan_to_spec(self._spec_path(idea_dir), output)

        # Append log entry
        append_log_entry(
            self._log_path(idea_dir),
            step=3,
            action="Experiment plan generated and code scaffolded",
            reasoning=(
                f"Plan: {output['plan_summary'][:200]}  |  "
                f"Estimated runtime: {output['expected_runtime_hours']}h  |  "
                f"Recommended backend: {output['compute_recommendation']}"
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "plan_summary": "Run proposed method vs baseline, compare metrics.",
        "datasets": ["CIFAR-10"],
        "baselines": ["Baseline"],
        "hyperparameters": {},
        "experiment_code": _FALLBACK_CODE,
        "requirements": ["torch>=2.0", "numpy", "scipy"],
        "expected_runtime_hours": 1.0,
        "compute_recommendation": "local",
    }
    out = {**defaults, **raw}
    if not out.get("experiment_code", "").strip():
        out["experiment_code"] = _FALLBACK_CODE
    return out


def _append_plan_to_spec(spec_path: str, plan: Dict[str, Any]) -> None:
    section = textwrap.dedent(f"""
        ## 5. Experimental Plan

        {plan.get('plan_summary', '')}

        ### Datasets
        {', '.join(plan.get('datasets', []))}

        ### Baselines
        {', '.join(plan.get('baselines', []))}

        ### Hyperparameter Grid
        {plan.get('hyperparameters', {})}

        ### Compute
        - Recommended backend : {plan.get('compute_recommendation', 'local')}
        - Estimated runtime   : {plan.get('expected_runtime_hours', '?')} hours

        ## 6. Flow Diagram

        ```
        START
          │
          ▼
        [Step 1] Idea generation
          │
          ▼
        [Step 2] Hypothesis formulation
          │
          ▼
        [Step 3] Experiment planning + code generation
          │
          ▼
        [Step 4] Run experiments (backend: {plan.get('compute_recommendation', 'local')})
          │
          ▼
        [Step 5] Validate results (p < 0.05, metrics meet criteria)
          │ Pass                     Fail
          ▼                          ▼
        [Step 6] Write paper      Revise (→ Step 2 or Step 3)
          │
          ▼
        END
        ```
    """)
    existing = read_text(spec_path)
    if "## 5. Experimental Plan" in existing:
        return  # already present
    write_text(spec_path, existing + section)
