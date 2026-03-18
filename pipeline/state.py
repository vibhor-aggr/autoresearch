"""
state.py – Pipeline state management.

Each idea in the pipeline is represented as an :class:`IdeaState` object.
State is persisted to ``ideas/<slug>/pipeline_state.json`` so the pipeline
can be interrupted and resumed without losing progress.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """Ordered stages in the autonomous research pipeline."""
    IDEA_GENERATION  = "IDEA_GENERATION"
    HYPOTHESIS       = "HYPOTHESIS"
    EXPERIMENT_PLAN  = "EXPERIMENT_PLAN"
    EXPERIMENT_RUN   = "EXPERIMENT_RUN"
    VALIDATION       = "VALIDATION"
    PAPER_WRITING    = "PAPER_WRITING"
    DONE             = "DONE"
    ABANDONED        = "ABANDONED"


# Ordered sequence for stage progression
STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.IDEA_GENERATION,
    PipelineStage.HYPOTHESIS,
    PipelineStage.EXPERIMENT_PLAN,
    PipelineStage.EXPERIMENT_RUN,
    PipelineStage.VALIDATION,
    PipelineStage.PAPER_WRITING,
    PipelineStage.DONE,
]


class RevisionTarget(str, Enum):
    """Which stage to jump back to when a revision is needed."""
    IDEA        = "IDEA"
    HYPOTHESIS  = "HYPOTHESIS"
    EXPERIMENTS = "EXPERIMENTS"


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class IdeaState:
    """
    Complete state for a single research idea as it moves through the pipeline.

    This object is serialised to JSON and read back on resume, so every field
    must be JSON-serialisable.
    """
    idea_id: str
    """Zero-padded three-digit ID, e.g. ``"003"``."""
    slug: str
    """URL-safe short name derived from the idea title, e.g. ``"my-new-idea"``."""
    current_stage: PipelineStage
    """The stage that should run next (or the most-recently completed stage)."""
    revision_round: int = 0
    """Number of revision loops that have been executed for this idea."""
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    """Log of each revision: {round, stage, reason, target, timestamp}."""
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    """Keyed by stage name; stores the structured output returned by each stage."""
    created_at: str = ""
    updated_at: str = ""

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        now = _utc_now()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp."""
        self.updated_at = _utc_now()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_stage"] = self.current_stage.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IdeaState":
        d = dict(d)
        d["current_stage"] = PipelineStage(d["current_stage"])
        return cls(**d)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def advance_stage(self) -> None:
        """Move to the next stage in :data:`STAGE_ORDER`."""
        try:
            idx = STAGE_ORDER.index(self.current_stage)
        except ValueError:
            return
        if idx + 1 < len(STAGE_ORDER):
            self.current_stage = STAGE_ORDER[idx + 1]
        self.touch()

    def record_revision(self, reason: str, target: RevisionTarget) -> None:
        """Append a revision record and update ``current_stage`` accordingly."""
        self.revision_round += 1
        self.revision_history.append({
            "round": self.revision_round,
            "from_stage": self.current_stage.value,
            "reason": reason,
            "target": target.value,
            "timestamp": _utc_now(),
        })
        if target == RevisionTarget.IDEA:
            self.current_stage = PipelineStage.IDEA_GENERATION
        elif target == RevisionTarget.HYPOTHESIS:
            self.current_stage = PipelineStage.HYPOTHESIS
        else:
            self.current_stage = PipelineStage.EXPERIMENT_PLAN
        self.touch()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_file_path(ideas_dir: str, slug: str) -> str:
    """Return the canonical path for an idea's state JSON file."""
    return os.path.join(ideas_dir, f"idea-???-{slug}", "pipeline_state.json")


def load_state(state_file: str) -> Optional[IdeaState]:
    """Load an :class:`IdeaState` from *state_file*; return ``None`` if absent."""
    if not os.path.exists(state_file):
        return None
    with open(state_file, encoding="utf-8") as fh:
        return IdeaState.from_dict(json.load(fh))


def save_state(state: IdeaState, state_file: str) -> None:
    """Persist *state* to *state_file* (creates parent directories as needed)."""
    os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2)
