"""
paper_writing.py – Stage 6: Generate a conference-ready research paper.

Produces:
  - ``paper/paper.md``   – Markdown draft
  - ``paper/main.tex``   – LaTeX source (NeurIPS-compatible template)
  - ``paper/references.bib`` – BibTeX bibliography

Output stored in ``state.stage_outputs["PAPER_WRITING"]``:
  {
    "paper_md": str,
    "paper_tex": str,
    "references_bib": str,
    "venue": str,
  }
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import append_log_entry, write_text

if TYPE_CHECKING:
    from pipeline.state import IdeaState


_SYSTEM = """\
You are an expert AI/ML researcher and academic writer.
Write rigorous, formal conference papers.
Respond with valid JSON only. Place all paper content as string values inside the JSON.
"""

_USER_TEMPLATE = """\
Write a complete research paper in the style of a top ML conference (NeurIPS / ICML / ICLR).

Idea:
  Title       : {title}
  Description : {description}

Hypothesis:
  H₀: {h0}
  H₁: {h1}
  Metrics: {metrics}

Experimental Results:
  Baseline metric : {baseline_mean:.4f}
  Proposed metric : {proposed_mean:.4f}
  Improvement     : {improvement_pct:.1f}%
  p-value         : {p_value}
  Accuracy drop   : {acc_drop:.2f} pp
  Validation      : {'PASSED' if passed else 'FAILED'}

Summary: {summary}

Return a JSON object with exactly these keys:
  "paper_md"       : full Markdown paper (abstract, intro, method, experiments, results,
                     conclusion, references – complete and self-contained)
  "paper_tex"      : full LaTeX source using \\documentclass{{article}} with standard
                     amsmath/booktabs packages; include abstract, all sections, and a
                     \\bibliography{{references}} command
  "references_bib" : BibTeX entries for at least 5 referenced works
  "venue"          : target conference / journal (e.g. "NeurIPS 2026")
"""


class PaperWritingStage(BaseStage):
    stage_name = "PAPER_WRITING"

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        idea_out = state.stage_outputs.get("IDEA_GENERATION", {})
        hyp_out  = state.stage_outputs.get("HYPOTHESIS", {})
        run_out  = state.stage_outputs.get("EXPERIMENT_RUN", {})
        val_out  = state.stage_outputs.get("VALIDATION", {})

        user_msg = _USER_TEMPLATE.format(
            title=idea_out.get("title", "Untitled"),
            description=idea_out.get("description", ""),
            h0=hyp_out.get("h0", ""),
            h1=hyp_out.get("h1", ""),
            metrics=hyp_out.get("metrics", []),
            baseline_mean=float(val_out.get("primary_metric_baseline", 0.0)),
            proposed_mean=float(val_out.get("primary_metric_proposed", 0.0)),
            improvement_pct=float(val_out.get("improvement_pct", 0.0)),
            p_value=val_out.get("p_value"),
            acc_drop=float(val_out.get("accuracy_drop_pp", 0.0)),
            passed=val_out.get("passed", False),
            summary=val_out.get("summary", ""),
        )

        output = self.llm.complete_json(_SYSTEM, user_msg)
        output = _normalise(output, idea_out, hyp_out, val_out)

        paper_dir = self._paper_dir(idea_dir)
        os.makedirs(paper_dir, exist_ok=True)

        write_text(os.path.join(paper_dir, "paper.md"),       output["paper_md"])
        write_text(os.path.join(paper_dir, "main.tex"),       output["paper_tex"])
        write_text(os.path.join(paper_dir, "references.bib"), output["references_bib"])

        append_log_entry(
            self._log_path(idea_dir),
            step=6,
            action=f"Research paper written (target: {output['venue']})",
            reasoning=(
                f"Generated paper.md, main.tex, references.bib in "
                f"{paper_dir}. "
                f"Venue: {output['venue']}."
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(
    raw: Dict[str, Any],
    idea_out: Dict[str, Any],
    hyp_out:  Dict[str, Any],
    val_out:  Dict[str, Any],
) -> Dict[str, Any]:
    title = idea_out.get("title", "Untitled")
    if not raw.get("paper_md", "").strip():
        raw["paper_md"] = _fallback_md(title, hyp_out, val_out)
    if not raw.get("paper_tex", "").strip():
        raw["paper_tex"] = _fallback_tex(title)
    if not raw.get("references_bib", "").strip():
        raw["references_bib"] = _fallback_bib()
    raw.setdefault("venue", "NeurIPS 2026")
    return raw


def _fallback_md(title: str, hyp_out: Dict, val_out: Dict) -> str:
    return textwrap.dedent(f"""\
        # {title}

        ## Abstract

        We study the problem described in this research.
        Our proposed method achieves an improvement of
        {val_out.get('improvement_pct', 0):.1f}% over the baseline
        (p={val_out.get('p_value')}).

        ## 1. Introduction

        *(Introduction placeholder – to be completed.)*

        ## 2. Method

        *(Method description – to be completed.)*

        ## 3. Experiments

        **H₀**: {hyp_out.get('h0', '')}

        **H₁**: {hyp_out.get('h1', '')}

        ## 4. Results

        | Metric | Baseline | Proposed | Δ |
        |---|---|---|---|
        | Primary | {val_out.get('primary_metric_baseline', ''):.4f} | {val_out.get('primary_metric_proposed', ''):.4f} | {val_out.get('improvement_pct', 0):.1f}% |

        p-value: {val_out.get('p_value')}

        ## 5. Conclusion

        *(Conclusion – to be completed.)*

        ## References

        *(References – to be completed.)*
    """)


def _fallback_tex(title: str) -> str:
    return textwrap.dedent(rf"""\
        \documentclass{{article}}
        \usepackage{{amsmath, booktabs, hyperref}}
        \title{{{title}}}
        \author{{Autonomous Researcher}}
        \date{{\today}}
        \begin{{document}}
        \maketitle
        \begin{{abstract}}
        Placeholder abstract.
        \end{{abstract}}
        \section{{Introduction}}
        Placeholder introduction.
        \section{{Method}}
        Placeholder method.
        \section{{Experiments}}
        Placeholder experiments.
        \section{{Results}}
        Placeholder results.
        \section{{Conclusion}}
        Placeholder conclusion.
        \bibliography{{references}}
        \bibliographystyle{{plainnat}}
        \end{{document}}
    """)


def _fallback_bib() -> str:
    return textwrap.dedent("""\
        @inproceedings{hinton2015distilling,
          title={Distilling the knowledge in a neural network},
          author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
          booktitle={NeurIPS Deep Learning Workshop},
          year={2015}
        }
    """)
