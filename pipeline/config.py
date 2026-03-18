"""
config.py – Configuration dataclasses and YAML loader for the pipeline.

Usage:
    cfg = load_config()                        # uses config/pipeline.yaml
    cfg = load_config("my_config.yaml")        # custom path
    cfg = PipelineConfig()                     # all defaults (stub LLM, local backend)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for the LLM provider used by all pipeline stages."""
    provider: str = "stub"
    """One of: ``openai``, ``anthropic``, ``stub`` (testing, no API calls)."""
    model: str = "gpt-4o"
    """Model identifier understood by the provider (e.g. ``gpt-4o``, ``claude-3-5-sonnet-20241022``)."""
    api_key_env: str = "OPENAI_API_KEY"
    """Environment variable that holds the API key."""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class BackendConfig:
    """Configuration for the compute backend used to run experiments."""
    type: str = "local"
    """One of: ``local``, ``kaggle``, ``remote``."""

    # ---- Kaggle ----
    kaggle_username: str = ""
    kaggle_key_env: str = "KAGGLE_KEY"
    """Environment variable that holds the Kaggle API key."""

    # ---- Remote SSH ----
    remote_host: str = ""
    remote_user: str = "ubuntu"
    remote_key_file: str = "~/.ssh/id_rsa"
    remote_work_dir: str = "/home/ubuntu/autoresearch"
    remote_timeout: int = 3600
    """Seconds to wait for a remote experiment to finish."""
    remote_known_hosts: str = "~/.ssh/known_hosts"
    """Path to the SSH known_hosts file used to verify the remote host identity.
    Set to an empty string to skip verification (insecure; not recommended)."""


@dataclass
class ValidationConfig:
    """Thresholds that determine whether a hypothesis is accepted."""
    significance_threshold: float = 0.05
    """p-value threshold for the paired t-test."""
    max_accuracy_drop_pp: float = 0.5
    """Maximum tolerable drop in Top-1 accuracy (percentage points)."""
    max_revision_rounds: int = 3
    """How many times to revise before abandoning an idea."""


@dataclass
class PipelineConfig:
    """Root configuration object for the entire pipeline."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    ideas_dir: str = "ideas"
    """Root directory where per-idea folders live (relative to repo root)."""
    master_log: str = "masterLog.md"
    """Path to the global master log markdown file."""
    continuous: bool = True
    """If True, keep generating new ideas after finishing each one."""
    max_ideas: int = 10
    """Maximum number of ideas to process (0 = unlimited)."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/pipeline.yaml") -> PipelineConfig:
    """
    Load a :class:`PipelineConfig` from *path*.

    Missing keys fall back to dataclass defaults.  If the file does not exist
    or PyYAML is not installed a default configuration is returned.
    """
    if not os.path.exists(path) or not _YAML_AVAILABLE:
        return PipelineConfig()

    with open(path, encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}

    def _build(cls, d):
        """Build a dataclass from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in (d or {}).items() if k in valid}
        return cls(**filtered)

    return PipelineConfig(
        llm=_build(LLMConfig, data.get("llm")),
        backend=_build(BackendConfig, data.get("backend")),
        validation=_build(ValidationConfig, data.get("validation")),
        ideas_dir=data.get("pipeline", {}).get("ideas_dir", "ideas"),
        master_log=data.get("pipeline", {}).get("master_log", "masterLog.md"),
        continuous=data.get("pipeline", {}).get("continuous", True),
        max_ideas=data.get("pipeline", {}).get("max_ideas", 10),
    )
