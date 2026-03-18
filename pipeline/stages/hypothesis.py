"""
hypothesis.py – Stage 2: Formulate a testable hypothesis.

Given the idea from Stage 1, the LLM is asked to produce:
  - A null hypothesis (H₀)
  - An alternative hypothesis (H₁)
  - Pre-registered acceptance criteria (metrics + thresholds)
  - A list of evaluation metrics

Output stored in ``state.stage_outputs["HYPOTHESIS"]``:
  {
    "research_question": str,
    "h0": str,
    "h1": str,
    "acceptance_criteria": {
        "p_value": float,          # e.g. 0.05
        "min_improvement": float,  # e.g. 5.0  (percentage)
        "max_acc_drop_pp": float,  # e.g. 0.5
    },
    "metrics": [str, ...],
    "min_runs": int,               # minimum independent runs
  }
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import append_log_entry, read_text, write_text

if TYPE_CHECKING:
    from pipeline.state import IdeaState


_SYSTEM = """\
You are a rigorous AI/ML researcher who pre-registers hypotheses to avoid p-hacking.
Respond with valid JSON only – no prose outside JSON.
"""

_USER_TEMPLATE = """\
Idea:
  Title       : {title}
  Description : {description}
  Problem     : {problem_statement}

Formulate a testable hypothesis for this idea.
Return a JSON object with exactly these keys:
  "research_question"   : one-sentence research question
  "h0"                  : null hypothesis (no effect / no improvement)
  "h1"                  : alternative hypothesis (the expected improvement)
  "acceptance_criteria" : object with keys:
                            "p_value"         (float, e.g. 0.05)
                            "min_improvement" (float %, relative to baseline metric)
                            "max_acc_drop_pp" (float, tolerable accuracy drop in pp)
  "metrics"             : list of metric strings (e.g. ["ECE", "Top-1 Acc", "NLL"])
  "min_runs"            : integer, minimum independent runs needed (usually 5)
"""


class HypothesisStage(BaseStage):
    stage_name = "HYPOTHESIS"

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        idea_out = state.stage_outputs.get("IDEA_GENERATION", {})

        user_msg = _USER_TEMPLATE.format(
            title=idea_out.get("title", ""),
            description=idea_out.get("description", ""),
            problem_statement=idea_out.get("problem_statement", ""),
        )

        output = self.llm.complete_json(_SYSTEM, user_msg)
        output = _normalise(output)

        # Append hypothesis section to specification.md
        _append_hypothesis_to_spec(self._spec_path(idea_dir), output)

        # Append log entry
        append_log_entry(
            self._log_path(idea_dir),
            step=2,
            action="Testable hypothesis formulated",
            reasoning=(
                f"H₀: {output['h0']}  |  H₁: {output['h1']}  "
                f"Pre-registered criterion: p < {output['acceptance_criteria'].get('p_value', 0.05)}"
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "research_question": "Does the proposed method improve the baseline?",
        "h0": "The proposed method performs no better than the baseline.",
        "h1": "The proposed method significantly outperforms the baseline.",
        "acceptance_criteria": {
            "p_value": 0.05,
            "min_improvement": 5.0,
            "max_acc_drop_pp": 0.5,
        },
        "metrics": ["Primary Metric", "Accuracy"],
        "min_runs": 5,
    }
    out = {**defaults, **raw}
    if not isinstance(out.get("acceptance_criteria"), dict):
        out["acceptance_criteria"] = defaults["acceptance_criteria"]
    return out


def _append_hypothesis_to_spec(spec_path: str, h: Dict[str, Any]) -> None:
    ac = h.get("acceptance_criteria", {})
    section = textwrap.dedent(f"""
        ## 2. Research Question & Hypothesis

        **Research Question**: {h.get('research_question', '')}

        **H₀**: {h.get('h0', '')}

        **H₁**: {h.get('h1', '')}

        ### Acceptance Criteria (pre-registered)

        | Criterion | Value |
        |---|---|
        | Significance threshold (p) | {ac.get('p_value', 0.05)} |
        | Minimum improvement (%) | {ac.get('min_improvement', 5.0)} |
        | Max tolerable accuracy drop (pp) | {ac.get('max_acc_drop_pp', 0.5)} |
        | Minimum independent runs | {h.get('min_runs', 5)} |

        **Metrics**: {', '.join(h.get('metrics', []))}
    """)
    existing = read_text(spec_path)
    if "## 2. Research Question" in existing:
        return  # already present (revision path)
    write_text(spec_path, existing + section)
