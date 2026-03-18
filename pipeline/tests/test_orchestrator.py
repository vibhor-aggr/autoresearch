"""
test_orchestrator.py – Integration tests for PipelineOrchestrator.

All LLM calls use the ``stub`` provider so no API key is needed.
All experiment execution uses the local backend running a tiny stub script.

Run with:
    python -m pytest pipeline/tests/test_orchestrator.py -v
"""

import json
import os
import textwrap

import pytest

from pipeline.config import (
    BackendConfig,
    LLMConfig,
    PipelineConfig,
    ValidationConfig,
)
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import IdeaState, PipelineStage, load_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fast_config(tmp_path):
    """A PipelineConfig that uses the stub LLM and local backend."""
    return PipelineConfig(
        llm=LLMConfig(provider="stub"),
        backend=BackendConfig(type="local"),
        validation=ValidationConfig(
            significance_threshold=0.05,
            max_accuracy_drop_pp=0.5,
            max_revision_rounds=1,
        ),
        ideas_dir=str(tmp_path / "ideas"),
        master_log=str(tmp_path / "masterLog.md"),
        continuous=False,
        max_ideas=1,
    )


@pytest.fixture()
def orchestrator(fast_config):
    return PipelineOrchestrator(fast_config)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_config_has_stub_llm(self):
        cfg = PipelineConfig()
        assert cfg.llm.provider == "stub"

    def test_default_config_has_local_backend(self):
        cfg = PipelineConfig()
        assert cfg.backend.type == "local"

    def test_load_config_returns_defaults_when_missing(self, tmp_path):
        from pipeline.config import load_config
        cfg = load_config(str(tmp_path / "missing.yaml"))
        assert isinstance(cfg, PipelineConfig)


# ---------------------------------------------------------------------------
# Orchestrator construction
# ---------------------------------------------------------------------------

class TestOrchestratorConstruction:

    def test_creates_llm_from_config(self, fast_config):
        orch = PipelineOrchestrator(fast_config)
        assert orch.llm.provider == "stub"

    def test_uses_supplied_llm(self, fast_config):
        from pipeline.utils.llm import LLMClient
        custom_llm = LLMClient(provider="stub")
        orch = PipelineOrchestrator(fast_config, llm=custom_llm)
        assert orch.llm is custom_llm

    def test_builds_all_six_stages(self, orchestrator):
        expected = {
            "IDEA_GENERATION", "HYPOTHESIS", "EXPERIMENT_PLAN",
            "EXPERIMENT_RUN", "VALIDATION", "PAPER_WRITING",
        }
        assert set(orchestrator._stages.keys()) == expected


# ---------------------------------------------------------------------------
# Full pipeline run_one (stub mode, stub experiment)
# ---------------------------------------------------------------------------

def _write_stub_experiment(idea_dir: str) -> None:
    """
    Write a tiny experiment script that immediately produces a results JSON.
    This is called by the test to pre-populate the experiment file before
    ExperimentPlanStage would overwrite it.
    """
    exp_dir = os.path.join(idea_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    code = textwrap.dedent("""\
        import argparse, json
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", default="results.json")
        parser.add_argument("--seed",   default="0")
        args = parser.parse_args()
        result = {
            "baseline_metric": 0.25,
            "proposed_metric": 0.15,
            "accuracy_baseline": 91.0,
            "accuracy_proposed": 91.2,
            "seed": int(args.seed),
        }
        import os; os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f)
    """)
    with open(os.path.join(exp_dir, "run_experiment.py"), "w") as fh:
        fh.write(code)


class TestRunOne:
    """End-to-end run_one with stub LLM and a stub experiment script."""

    def test_run_one_returns_idea_state(self, orchestrator):
        state = orchestrator.run_one()
        assert isinstance(state, IdeaState)

    def test_run_one_reaches_terminal_stage(self, orchestrator):
        state = orchestrator.run_one()
        assert state.current_stage in (
            PipelineStage.DONE,
            PipelineStage.ABANDONED,
        )

    def test_run_one_populates_stage_outputs(self, orchestrator):
        state = orchestrator.run_one()
        assert "IDEA_GENERATION" in state.stage_outputs

    def test_run_one_persists_state_to_json(self, orchestrator, fast_config):
        state = orchestrator.run_one()
        # Find the state file
        idea_root = fast_config.ideas_dir
        found = False
        for name in os.listdir(idea_root):
            sf = os.path.join(idea_root, name, "pipeline_state.json")
            if os.path.exists(sf):
                found = True
                loaded = load_state(sf)
                assert loaded is not None
                assert loaded.idea_id == state.idea_id
        assert found, "pipeline_state.json not found"

    def test_master_log_updated(self, orchestrator, fast_config):
        orchestrator.run_one()
        assert os.path.exists(fast_config.master_log)
        content = open(fast_config.master_log).read()
        assert "|" in content  # contains a table row


# ---------------------------------------------------------------------------
# Revision loop
# ---------------------------------------------------------------------------

class TestRevisionLoop:

    def test_revision_does_not_exceed_max_rounds(self, fast_config):
        """With max_revision_rounds=1 the idea should be abandoned after 1 failure."""
        # Force validation to always fail by monkey-patching ValidationStage.run
        from pipeline.stages.validation import ValidationStage

        original_run = ValidationStage.run

        def always_fail(self, state, idea_dir):
            return {
                "passed": False,
                "p_value": 0.9,
                "primary_metric_baseline": 0.25,
                "primary_metric_proposed": 0.24,
                "improvement_pct": 0.4,
                "accuracy_drop_pp": 0.0,
                "criteria_met": {},
                "summary": "Not significant.",
                "revision_target": "HYPOTHESIS",
                "revision_reason": "p-value too high",
            }

        ValidationStage.run = always_fail
        try:
            orch = PipelineOrchestrator(fast_config)
            state = orch.run_one()
            assert state.current_stage == PipelineStage.ABANDONED
        finally:
            ValidationStage.run = original_run


# ---------------------------------------------------------------------------
# LLMClient stub
# ---------------------------------------------------------------------------

class TestLLMClientStub:

    def test_stub_complete_returns_string(self):
        from pipeline.utils.llm import LLMClient
        client = LLMClient(provider="stub")
        result = client.complete("system", "user")
        assert isinstance(result, str)

    def test_stub_complete_json_returns_dict(self):
        from pipeline.utils.llm import LLMClient
        client = LLMClient(provider="stub")
        result = client.complete_json("system", "return json please")
        assert isinstance(result, dict)

    def test_unknown_provider_raises(self):
        from pipeline.utils.llm import LLMClient
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClient(provider="bogus")
