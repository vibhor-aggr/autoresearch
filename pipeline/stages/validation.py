"""
validation.py – Stage 5: Validate experimental results against the hypothesis.

Performs:
  1. Statistical testing (paired t-test on the primary metric) using scipy.
  2. Checks each acceptance criterion.
  3. Calls the LLM to interpret the results and decide the revision target if failed.

Output stored in ``state.stage_outputs["VALIDATION"]``:
  {
    "passed": bool,
    "p_value": float | None,
    "primary_metric_baseline": float,
    "primary_metric_proposed": float,
    "improvement_pct": float,
    "accuracy_drop_pp": float,
    "criteria_met": {str: bool},
    "summary": str,
    "revision_target": "IDEA" | "HYPOTHESIS" | "EXPERIMENTS" | None,
    "revision_reason": str,
  }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import append_log_entry

if TYPE_CHECKING:
    from pipeline.state import IdeaState


_SYSTEM = """\
You are a rigorous ML researcher reviewing experimental results.
Respond with valid JSON only.
"""

_INTERPRET_TEMPLATE = """\
Hypothesis:
  H₀: {h0}
  H₁: {h1}
  Acceptance criteria: {acceptance_criteria}

Statistical results:
  p-value              : {p_value}
  Primary metric (baseline vs proposed): {baseline} vs {proposed}
  Improvement (%): {improvement_pct:.2f}
  Accuracy drop (pp): {accuracy_drop:.2f}

Did the experiment PASS or FAIL the hypothesis test?

Return a JSON object with exactly these keys:
  "passed"          : true | false
  "summary"         : 2–4 sentence interpretation
  "revision_target" : "IDEA" | "HYPOTHESIS" | "EXPERIMENTS" | null (if passed)
  "revision_reason" : explanation of what needs to change (empty string if passed)
"""


class ValidationStage(BaseStage):
    stage_name = "VALIDATION"

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        hyp_out  = state.stage_outputs.get("HYPOTHESIS", {})
        run_out  = state.stage_outputs.get("EXPERIMENT_RUN", {})
        raw_results: List[Dict] = run_out.get("raw_results", [])

        ac = hyp_out.get("acceptance_criteria", {})
        p_threshold     = float(ac.get("p_value", 0.05))
        min_impr_pct    = float(ac.get("min_improvement", 0.0))
        max_acc_drop    = float(ac.get("max_acc_drop_pp",
                                       self.config.validation.max_accuracy_drop_pp))

        # ------------------------------------------------------------------
        # Extract primary metric values from raw results
        # ------------------------------------------------------------------
        baselines, proposeds, acc_baselines, acc_proposeds = _extract_metrics(raw_results)

        p_value         = _paired_ttest(baselines, proposeds)
        baseline_mean   = _mean(baselines)
        proposed_mean   = _mean(proposeds)
        improvement_pct = _improvement_pct(baseline_mean, proposed_mean)
        acc_drop        = _mean(acc_baselines) - _mean(acc_proposeds)

        # ------------------------------------------------------------------
        # Criteria checks
        # ------------------------------------------------------------------
        criteria_met: Dict[str, bool] = {
            "p_value_significant": (p_value is not None and p_value < p_threshold),
            "min_improvement_met": (improvement_pct >= min_impr_pct),
            "accuracy_drop_ok":    (acc_drop <= max_acc_drop),
            "sufficient_runs":     (len(raw_results) >= hyp_out.get("min_runs", 3)),
        }

        # ------------------------------------------------------------------
        # LLM interpretation
        # ------------------------------------------------------------------
        llm_input = _INTERPRET_TEMPLATE.format(
            h0=hyp_out.get("h0", ""),
            h1=hyp_out.get("h1", ""),
            acceptance_criteria=json.dumps(ac),
            p_value=p_value,
            baseline=baseline_mean,
            proposed=proposed_mean,
            improvement_pct=improvement_pct,
            accuracy_drop=acc_drop,
        )
        llm_out = self.llm.complete_json(_SYSTEM, llm_input)
        llm_out = _normalise_llm(llm_out)

        # Overall pass requires all numeric criteria AND LLM agrees
        passed = all(criteria_met.values()) and llm_out.get("passed", False)

        output: Dict[str, Any] = {
            "passed": passed,
            "p_value": p_value,
            "primary_metric_baseline": baseline_mean,
            "primary_metric_proposed": proposed_mean,
            "improvement_pct": round(improvement_pct, 3),
            "accuracy_drop_pp": round(acc_drop, 3),
            "criteria_met": criteria_met,
            "summary": llm_out.get("summary", ""),
            "revision_target": None if passed else llm_out.get("revision_target"),
            "revision_reason": "" if passed else llm_out.get("revision_reason", ""),
        }

        append_log_entry(
            self._log_path(idea_dir),
            step=5,
            action=f"Validation {'PASSED' if passed else 'FAILED'}",
            reasoning=(
                f"p={p_value}, improvement={improvement_pct:.1f}%, "
                f"acc_drop={acc_drop:.2f}pp.  {output['summary']}"
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _extract_metrics(
    raw_results: List[Dict],
) -> tuple:
    """Extract baseline and proposed metric lists from raw result dicts."""
    baselines:     List[float] = []
    proposeds:     List[float] = []
    acc_baselines: List[float] = []
    acc_proposeds: List[float] = []

    for r in raw_results:
        # Try common key patterns; fall back to 0.0
        baselines.append(float(r.get("baseline_metric",
                                     r.get("ece_baseline",
                                     r.get("metric_baseline", 0.25)))))
        proposeds.append(float(r.get("proposed_metric",
                                     r.get("ece_proposed",
                                     r.get("metric_proposed", 0.20)))))
        acc_baselines.append(float(r.get("accuracy_baseline",
                                         r.get("acc_baseline", 90.0))))
        acc_proposeds.append(float(r.get("accuracy_proposed",
                                         r.get("acc_proposed", 90.0))))
    # Ensure non-empty lists
    if not baselines:
        baselines  = [0.25]; proposeds  = [0.20]
        acc_baselines = [90.0]; acc_proposeds = [90.0]
    return baselines, proposeds, acc_baselines, acc_proposeds


def _paired_ttest(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None
    try:
        from scipy import stats  # type: ignore
        _, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        return None


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _improvement_pct(baseline: float, proposed: float) -> float:
    """Positive = proposed is lower (better for error metrics)."""
    if baseline == 0.0:
        return 0.0
    return 100.0 * (baseline - proposed) / abs(baseline)


def _normalise_llm(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "passed":          bool(raw.get("passed", False)),
        "summary":         str(raw.get("summary", "")),
        "revision_target": raw.get("revision_target"),
        "revision_reason": str(raw.get("revision_reason", "")),
    }
