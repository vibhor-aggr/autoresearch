"""
base.py – Abstract base class for pipeline stages.

Every stage:
  1. Receives the current :class:`IdeaState` and configuration.
  2. Calls the LLM (or executes code) to produce a structured output.
  3. Updates ``state.stage_outputs[stage_name]``.
  4. Advances (or redirects) ``state.current_stage``.
  5. Persists state and updates Markdown logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig
    from pipeline.state import IdeaState
    from pipeline.utils.llm import LLMClient


class BaseStage(ABC):
    """
    Abstract base for all pipeline stages.

    Subclasses must implement :meth:`run` and expose :attr:`stage_name`.
    """

    #: Unique string key used in ``state.stage_outputs``
    stage_name: str = ""

    def __init__(self, config: "PipelineConfig", llm: "LLMClient") -> None:
        self.config = config
        self.llm = llm

    @abstractmethod
    def run(self, state: "IdeaState", idea_dir: str) -> Dict[str, Any]:
        """
        Execute the stage.

        Parameters
        ----------
        state:
            Current pipeline state for this idea.
        idea_dir:
            Absolute path to the idea's root directory.

        Returns
        -------
        dict
            Structured output that will be stored in
            ``state.stage_outputs[self.stage_name]``.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers available to all stages
    # ------------------------------------------------------------------

    def _log_path(self, idea_dir: str) -> str:
        return f"{idea_dir}/log.md"

    def _spec_path(self, idea_dir: str) -> str:
        return f"{idea_dir}/specification.md"

    def _experiments_dir(self, idea_dir: str) -> str:
        return f"{idea_dir}/experiments"

    def _paper_dir(self, idea_dir: str) -> str:
        return f"{idea_dir}/paper"
