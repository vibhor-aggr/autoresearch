"""
idea_generation.py – Stage 1: Generate a novel, tractable research idea.

The LLM is asked to propose an idea from the AI/ML space that:
  - is novel (not present in the list of existing ideas)
  - is tractable (can be tested with standard datasets / compute)
  - has a clear research question

Output stored in ``state.stage_outputs["IDEA_GENERATION"]``:
  {
    "title": str,
    "slug": str,           # URL-safe short name
    "description": str,    # one-sentence description
    "problem_statement": str,
    "novelty_assessment": str,
    "novelty_score": int,  # 1–10
    "venue_target": str,   # e.g. "NeurIPS / ICML"
  }
"""

from __future__ import annotations

import re
from typing import Any, Dict, TYPE_CHECKING

from pipeline.stages.base import BaseStage
from pipeline.utils.file_ops import (
    append_log_entry,
    existing_ideas_summary,
    write_text,
)

if TYPE_CHECKING:
    from pipeline.state import IdeaState


_SYSTEM = """\
You are an expert AI/ML researcher with deep knowledge of the literature.
Your task is to identify novel, tractable research problems.
A good idea:
  - Is not already well-solved in the literature.
  - Can be tested with standard datasets (CIFAR, ImageNet, GLUE, etc.) on a single GPU.
  - Has a clear, testable research question.
  - Would be publishable at NeurIPS, ICML, ICLR, ACL, or similar.
Always respond with valid JSON only – no prose, no code fences outside JSON.
"""

_USER_TEMPLATE = """\
Existing ideas (avoid duplicates):
{existing}

Propose ONE new AI/ML research idea.
Return a JSON object with these exact keys:
  "title"              : short human-readable title (≤ 10 words)
  "slug"               : URL-safe lowercase slug (hyphens, no spaces)
  "description"        : one-sentence description (≤ 25 words)
  "problem_statement"  : 2–4 sentences explaining the problem and why it matters
  "novelty_assessment" : 2–4 sentences explaining why the idea is novel
  "novelty_score"      : integer 1–10 (10 = completely novel)
  "venue_target"       : e.g. "NeurIPS / ICML"
"""


class IdeaGenerationStage(BaseStage):
    stage_name = "IDEA_GENERATION"

    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        existing = existing_ideas_summary(self.config.ideas_dir)
        user_msg = _USER_TEMPLATE.format(existing=existing)

        output = self.llm.complete_json(_SYSTEM, user_msg)
        output = _normalise_output(output)

        # Write specification.md
        spec = _build_specification(output)
        write_text(self._spec_path(idea_dir), spec)

        # Append to log.md
        append_log_entry(
            self._log_path(idea_dir),
            step=1,
            action=f"Idea generated: {output['title']}",
            reasoning=(
                f"Novelty score: {output.get('novelty_score', '?')}/10. "
                f"{output.get('novelty_assessment', '')}"
            ),
        )

        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist and slug is URL-safe."""
    defaults = {
        "title": "Untitled Idea",
        "slug": "untitled-idea",
        "description": "",
        "problem_statement": "",
        "novelty_assessment": "",
        "novelty_score": 5,
        "venue_target": "TBD",
    }
    out = {**defaults, **raw}
    # Sanitise slug
    out["slug"] = re.sub(r"[^a-z0-9]+", "-", out["slug"].lower()).strip("-")
    return out


def _build_specification(output: Dict[str, Any]) -> str:
    return f"""\
# Specification: {output['title']}

## 1. Problem Statement

{output.get('problem_statement', '')}

## 2. Novelty Assessment

{output.get('novelty_assessment', '')}

**Novelty Score**: {output.get('novelty_score', '?')}/10

## 3. Venue Target

{output.get('venue_target', 'TBD')}

## 4. Research Question

*(To be filled in by HypothesisStage)*

## 5. Experimental Plan

*(To be filled in by ExperimentPlanStage)*

## 6. Flow Diagram

*(To be filled in by ExperimentPlanStage)*
"""
