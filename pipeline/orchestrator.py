"""
orchestrator.py – PipelineOrchestrator: main autonomous research loop.

The orchestrator drives the state machine:

    IDEA_GENERATION → HYPOTHESIS → EXPERIMENT_PLAN → EXPERIMENT_RUN
        → VALIDATION → PAPER_WRITING → DONE

On validation failure it redirects:
    VALIDATION → { IDEA_GENERATION | HYPOTHESIS | EXPERIMENT_PLAN }
                   (as determined by the LLM)

It also:
  - Scaffolds the per-idea directory tree
  - Persists :class:`IdeaState` as JSON after every stage
  - Updates ``masterLog.md`` with current status
  - Respects ``max_revision_rounds`` before abandoning an idea
  - Loops continuously (if ``config.continuous=True``) until ``max_ideas`` reached

Usage::

    from pipeline import PipelineOrchestrator, load_config
    cfg = load_config()
    orch = PipelineOrchestrator(cfg)
    orch.run()           # continuous loop
    orch.run_one()       # single idea end-to-end
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pipeline.config import PipelineConfig
from pipeline.state import (
    IdeaState,
    PipelineStage,
    RevisionTarget,
    load_state,
    save_state,
)
from pipeline.utils.file_ops import (
    next_idea_id,
    scaffold_idea_directory,
    update_master_log_status,
)
from pipeline.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Drives the autonomous AI/ML research pipeline.

    Parameters
    ----------
    config:
        A :class:`PipelineConfig` instance.
    llm:
        Optional pre-built :class:`LLMClient`.  If ``None``, one is created
        from ``config.llm``.
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.llm = llm or LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key_env=config.llm.api_key_env,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        self._stages = self._build_stages()
        self._ideas_processed = 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the pipeline loop continuously.

        Stops when:
          - ``config.max_ideas > 0`` and that many ideas have been completed/abandoned.
          - A :exc:`KeyboardInterrupt` is received.
        """
        logger.info("Pipeline started.  continuous=%s  max_ideas=%s",
                    self.config.continuous, self.config.max_ideas)
        try:
            while True:
                self.run_one()
                self._ideas_processed += 1
                limit = self.config.max_ideas
                if limit > 0 and self._ideas_processed >= limit:
                    logger.info("Reached max_ideas=%d – stopping.", limit)
                    break
                if not self.config.continuous:
                    break
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")

    def run_one(self) -> IdeaState:
        """
        Run the pipeline for one idea from start to finish (or abandonment).

        Returns the final :class:`IdeaState`.
        """
        # Allocate a new idea
        idea_id  = next_idea_id(self.config.ideas_dir)
        slug     = "tbd"          # overwritten by IdeaGenerationStage
        idea_dir = scaffold_idea_directory(
            self.config.ideas_dir, idea_id, slug
        )
        state_file = os.path.join(idea_dir, "pipeline_state.json")

        # Reuse existing state if present (resume)
        state = load_state(state_file) or IdeaState(
            idea_id=idea_id,
            slug=slug,
            current_stage=PipelineStage.IDEA_GENERATION,
        )

        self._update_master_log(state, "PROPOSED")
        logger.info("Starting idea %s (stage: %s)", state.idea_id, state.current_stage)

        while state.current_stage not in (PipelineStage.DONE, PipelineStage.ABANDONED):
            stage_name = state.current_stage.value
            stage = self._stages.get(stage_name)
            if stage is None:
                logger.warning("No stage handler for '%s' – skipping.", stage_name)
                state.advance_stage()
                save_state(state, state_file)
                continue

            # Resolve the actual idea_dir (slug may have been updated)
            idea_dir = self._resolve_idea_dir(state)
            state_file = os.path.join(idea_dir, "pipeline_state.json")

            logger.info("[%s] Running stage %s", state.idea_id, stage_name)
            try:
                output = stage.run(state, idea_dir)
            except Exception as exc:
                logger.error("[%s] Stage %s failed: %s", state.idea_id, stage_name, exc)
                self._handle_stage_failure(state, stage_name, str(exc))
                save_state(state, state_file)
                self._update_master_log(state, "FAILED")
                break

            # Store output and advance
            state.stage_outputs[stage_name] = output
            self._post_stage(state, stage_name, output)
            save_state(state, state_file)

        logger.info("[%s] Finished with stage %s", state.idea_id, state.current_stage)
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_stages(self) -> dict:
        """Instantiate all stage objects keyed by their stage_name."""
        from pipeline.stages.idea_generation import IdeaGenerationStage
        from pipeline.stages.hypothesis      import HypothesisStage
        from pipeline.stages.experiment_plan import ExperimentPlanStage
        from pipeline.stages.experiment_run  import ExperimentRunStage
        from pipeline.stages.validation      import ValidationStage
        from pipeline.stages.paper_writing   import PaperWritingStage

        return {
            "IDEA_GENERATION":  IdeaGenerationStage(self.config, self.llm),
            "HYPOTHESIS":       HypothesisStage(self.config, self.llm),
            "EXPERIMENT_PLAN":  ExperimentPlanStage(self.config, self.llm),
            "EXPERIMENT_RUN":   ExperimentRunStage(self.config, self.llm),
            "VALIDATION":       ValidationStage(self.config, self.llm),
            "PAPER_WRITING":    PaperWritingStage(self.config, self.llm),
        }

    def _post_stage(
        self,
        state: IdeaState,
        stage_name: str,
        output: dict,
    ) -> None:
        """
        Handle post-stage logic: update slug, check validation, advance stage.
        """
        # After idea generation: update slug and rename directory
        if stage_name == "IDEA_GENERATION":
            new_slug = output.get("slug", state.slug)
            if new_slug != state.slug:
                state.slug = new_slug
            self._update_master_log(state, "NOVELTY_OK")

        elif stage_name == "HYPOTHESIS":
            self._update_master_log(state, "HYPOTHESIS")

        elif stage_name == "EXPERIMENT_PLAN":
            self._update_master_log(state, "EXPERIMENTS")

        elif stage_name == "EXPERIMENT_RUN":
            self._update_master_log(state, "RESULTS")

        elif stage_name == "VALIDATION":
            if output.get("passed"):
                self._update_master_log(state, "VALIDATED")
                state.advance_stage()  # → PAPER_WRITING
            else:
                self._handle_validation_failure(state, output)
            return  # advance already handled

        elif stage_name == "PAPER_WRITING":
            self._update_master_log(state, "WRITING")
            state.advance_stage()  # → DONE
            self._update_master_log(state, "SUBMITTED")
            return

        state.advance_stage()

    def _handle_validation_failure(self, state: IdeaState, val_output: dict) -> None:
        """Decide revision target and loop back, or abandon after too many rounds."""
        max_rounds = self.config.validation.max_revision_rounds
        if state.revision_round >= max_rounds:
            logger.warning(
                "[%s] Max revision rounds (%d) reached – abandoning.",
                state.idea_id, max_rounds,
            )
            state.current_stage = PipelineStage.ABANDONED
            self._update_master_log(state, "ABANDONED")
            return

        raw_target   = val_output.get("revision_target", "EXPERIMENTS")
        reason       = val_output.get("revision_reason", "Validation failed.")
        target_map   = {
            "IDEA":        RevisionTarget.IDEA,
            "HYPOTHESIS":  RevisionTarget.HYPOTHESIS,
            "EXPERIMENTS": RevisionTarget.EXPERIMENTS,
        }
        target = target_map.get(str(raw_target).upper(), RevisionTarget.EXPERIMENTS)
        logger.info(
            "[%s] Validation failed – revising toward %s. Reason: %s",
            state.idea_id, target, reason,
        )
        state.record_revision(reason, target)

    def _handle_stage_failure(
        self, state: IdeaState, stage_name: str, error_msg: str
    ) -> None:
        """Mark the idea as abandoned after an unrecoverable stage error."""
        state.current_stage = PipelineStage.ABANDONED
        state.revision_history.append({
            "round": state.revision_round,
            "from_stage": stage_name,
            "reason": f"Unrecoverable error: {error_msg}",
            "target": "ABANDONED",
        })
        state.touch()

    def _resolve_idea_dir(self, state: IdeaState) -> str:
        """Return the idea directory path using the current state slug."""
        from pipeline.utils.file_ops import idea_dir as _idea_dir
        base = _idea_dir(self.config.ideas_dir, state.idea_id, state.slug)
        if not os.path.exists(base):
            # Try to find it by scanning (slug may differ from directory name)
            root = self.config.ideas_dir
            if os.path.exists(root):
                for name in os.listdir(root):
                    if name.startswith(f"idea-{state.idea_id}-"):
                        base = os.path.join(root, name)
                        break
            os.makedirs(base, exist_ok=True)
        return base

    def _update_master_log(self, state: IdeaState, status: str) -> None:
        idea_out = state.stage_outputs.get("IDEA_GENERATION", {})
        update_master_log_status(
            master_log_path=self.config.master_log,
            idea_id=state.idea_id,
            slug=state.slug,
            description=idea_out.get("description", state.slug),
            status=status,
            venue=idea_out.get("venue_target", "TBD"),
        )
