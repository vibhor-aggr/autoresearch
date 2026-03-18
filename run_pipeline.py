"""
run_pipeline.py – CLI entry point for the autonomous research pipeline.

Usage
-----
# Run with default config (stub LLM + local backend, 1 idea):
    python run_pipeline.py

# Run continuously (real LLM, Kaggle backend):
    python run_pipeline.py --config config/pipeline.yaml

# Run a single idea then stop:
    python run_pipeline.py --config config/pipeline.yaml --max-ideas 1

# Override LLM provider / backend on the command line:
    python run_pipeline.py --llm-provider openai --backend local

Environment variables
---------------------
OPENAI_API_KEY   – required when --llm-provider openai
ANTHROPIC_API_KEY – required when --llm-provider anthropic
KAGGLE_USERNAME  – required when --backend kaggle
KAGGLE_KEY       – required when --backend kaggle
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args():
    p = argparse.ArgumentParser(
        description="Autonomous AI/ML research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default="config/pipeline.yaml",
        help="Path to pipeline YAML config (default: config/pipeline.yaml)",
    )
    p.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "stub"],
        default=None,
        help="Override the LLM provider from the config file",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Override the LLM model from the config file (e.g. gpt-4o)",
    )
    p.add_argument(
        "--backend",
        choices=["local", "kaggle", "remote"],
        default=None,
        help="Override the compute backend from the config file",
    )
    p.add_argument(
        "--max-ideas",
        type=int,
        default=None,
        help="Maximum number of ideas to process (0 = unlimited)",
    )
    p.add_argument(
        "--ideas-dir",
        default=None,
        help="Override the ideas directory path",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from pipeline.config import load_config
    from pipeline.orchestrator import PipelineOrchestrator

    cfg = load_config(args.config)

    # Command-line overrides
    if args.llm_provider is not None:
        cfg.llm.provider = args.llm_provider
    if args.llm_model is not None:
        cfg.llm.model = args.llm_model
    if args.backend is not None:
        cfg.backend.type = args.backend
    if args.max_ideas is not None:
        cfg.max_ideas = args.max_ideas
    if args.ideas_dir is not None:
        cfg.ideas_dir = args.ideas_dir

    orch = PipelineOrchestrator(cfg)
    orch.run()


if __name__ == "__main__":
    main()
