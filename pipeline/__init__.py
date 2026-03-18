"""
pipeline – Autonomous AI/ML research pipeline.

Entry points:
  run_pipeline.py          CLI entry point
  PipelineOrchestrator     main loop class

Stages (in order):
  1. IdeaGenerationStage   – generate novel research ideas via LLM
  2. HypothesisStage       – formulate testable H₀/H₁ pairs
  3. ExperimentPlanStage   – plan and generate experiment code
  4. ExperimentRunStage    – execute experiments on chosen backend
  5. ValidationStage       – statistical validation; decide pass/fail/revise
  6. PaperWritingStage     – generate Markdown + LaTeX paper

Backends:
  LocalBackend             – run experiments in a local subprocess
  KaggleBackend            – submit to Kaggle Notebooks via Kaggle API
  RemoteSSHBackend         – SSH into a remote server to run experiments
"""

from pipeline.config import PipelineConfig, load_config
from pipeline.state import IdeaState, PipelineStage, RevisionTarget
from pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineConfig",
    "load_config",
    "IdeaState",
    "PipelineStage",
    "RevisionTarget",
    "PipelineOrchestrator",
]
