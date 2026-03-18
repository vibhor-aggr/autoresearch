"""
test_state.py – Unit tests for pipeline.state

Run with:
    python -m pytest pipeline/tests/test_state.py -v
"""

import json
import os
import tempfile

import pytest

from pipeline.state import (
    IdeaState,
    PipelineStage,
    RevisionTarget,
    STAGE_ORDER,
    load_state,
    save_state,
)


# ---------------------------------------------------------------------------
# IdeaState construction
# ---------------------------------------------------------------------------

class TestIdeaStateConstruction:

    def test_defaults_set(self):
        s = IdeaState(idea_id="001", slug="test-idea",
                      current_stage=PipelineStage.IDEA_GENERATION)
        assert s.revision_round == 0
        assert s.revision_history == []
        assert s.stage_outputs == {}
        assert s.created_at != ""
        assert s.updated_at != ""

    def test_timestamps_are_utc_format(self):
        s = IdeaState(idea_id="001", slug="test",
                      current_stage=PipelineStage.IDEA_GENERATION)
        # Format: YYYY-MM-DDTHH:MM:SSZ
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s.created_at)


# ---------------------------------------------------------------------------
# Stage progression
# ---------------------------------------------------------------------------

class TestStageProgression:

    def test_advance_moves_to_next_stage(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.IDEA_GENERATION)
        s.advance_stage()
        assert s.current_stage == PipelineStage.HYPOTHESIS

    def test_advance_full_sequence(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.IDEA_GENERATION)
        expected = STAGE_ORDER[1:]
        for exp in expected:
            s.advance_stage()
            assert s.current_stage == exp

    def test_advance_from_done_is_noop(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.DONE)
        s.advance_stage()
        assert s.current_stage == PipelineStage.DONE


# ---------------------------------------------------------------------------
# Revision
# ---------------------------------------------------------------------------

class TestRevision:

    def test_revision_increments_round(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.VALIDATION)
        s.record_revision("p-value too high", RevisionTarget.HYPOTHESIS)
        assert s.revision_round == 1

    def test_revision_updates_stage_hypothesis(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.VALIDATION)
        s.record_revision("need better H1", RevisionTarget.HYPOTHESIS)
        assert s.current_stage == PipelineStage.HYPOTHESIS

    def test_revision_updates_stage_idea(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.VALIDATION)
        s.record_revision("idea not novel", RevisionTarget.IDEA)
        assert s.current_stage == PipelineStage.IDEA_GENERATION

    def test_revision_updates_stage_experiments(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.VALIDATION)
        s.record_revision("need more runs", RevisionTarget.EXPERIMENTS)
        assert s.current_stage == PipelineStage.EXPERIMENT_PLAN

    def test_revision_history_appended(self):
        s = IdeaState(idea_id="001", slug="x",
                      current_stage=PipelineStage.VALIDATION)
        s.record_revision("reason A", RevisionTarget.HYPOTHESIS)
        s.record_revision("reason B", RevisionTarget.EXPERIMENTS)
        assert len(s.revision_history) == 2
        assert s.revision_history[0]["reason"] == "reason A"
        assert s.revision_history[1]["target"] == RevisionTarget.EXPERIMENTS.value


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisationRoundtrip:

    def _make_state(self) -> IdeaState:
        s = IdeaState(idea_id="007", slug="my-idea",
                      current_stage=PipelineStage.EXPERIMENT_RUN)
        s.stage_outputs["IDEA_GENERATION"] = {"title": "Test", "slug": "my-idea"}
        s.record_revision("needs work", RevisionTarget.EXPERIMENTS)
        return s

    def test_to_dict_current_stage_is_string(self):
        s = self._make_state()
        d = s.to_dict()
        assert isinstance(d["current_stage"], str)

    def test_from_dict_restores_stage_enum(self):
        s = self._make_state()
        restored = IdeaState.from_dict(s.to_dict())
        assert restored.current_stage == s.current_stage

    def test_roundtrip_preserves_all_fields(self):
        s = self._make_state()
        restored = IdeaState.from_dict(s.to_dict())
        assert restored.idea_id == s.idea_id
        assert restored.slug == s.slug
        assert restored.revision_round == s.revision_round
        assert restored.revision_history == s.revision_history
        assert restored.stage_outputs == s.stage_outputs


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------

class TestFilePersistence:

    def test_save_and_load_roundtrip(self, tmp_path):
        s = IdeaState(idea_id="002", slug="save-test",
                      current_stage=PipelineStage.HYPOTHESIS)
        s.stage_outputs["IDEA_GENERATION"] = {"title": "Saved Idea"}
        path = str(tmp_path / "state.json")
        save_state(s, path)
        loaded = load_state(path)
        assert loaded is not None
        assert loaded.idea_id == "002"
        assert loaded.current_stage == PipelineStage.HYPOTHESIS
        assert loaded.stage_outputs["IDEA_GENERATION"]["title"] == "Saved Idea"

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_state(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "state.json")
        s = IdeaState(idea_id="003", slug="deep",
                      current_stage=PipelineStage.DONE)
        save_state(s, path)
        assert os.path.exists(path)

    def test_saved_file_is_valid_json(self, tmp_path):
        s = IdeaState(idea_id="004", slug="json-check",
                      current_stage=PipelineStage.PAPER_WRITING)
        path = str(tmp_path / "s.json")
        save_state(s, path)
        with open(path) as f:
            data = json.load(f)
        assert data["idea_id"] == "004"
