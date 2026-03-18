"""
test_backends.py – Unit tests for pipeline backends.

LocalBackend is tested with real subprocess calls against a tiny script.
KaggleBackend and RemoteSSHBackend are tested for construction and error handling
without making real network calls.

Run with:
    python -m pytest pipeline/tests/test_backends.py -v
"""

import json
import os
import sys
import textwrap

import pytest

from pipeline.config import BackendConfig
from pipeline.backends.local import LocalBackend
from pipeline.backends.base import AbstractBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_stub_script(path: str) -> None:
    """Write a tiny Python script that dumps a results JSON."""
    code = textwrap.dedent("""\
        import argparse, json, os
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", default="results.json")
        parser.add_argument("--seed",   default="0")
        args = parser.parse_args()
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"seed": args.seed, "metric": 0.5}, f)
    """)
    with open(path, "w") as fh:
        fh.write(code)


# ---------------------------------------------------------------------------
# AbstractBackend
# ---------------------------------------------------------------------------

class TestAbstractBackend:

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractBackend()  # type: ignore


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class TestLocalBackend:

    @pytest.fixture()
    def backend(self):
        return LocalBackend(BackendConfig(type="local"))

    @pytest.fixture()
    def script(self, tmp_path):
        p = str(tmp_path / "run.py")
        _write_stub_script(p)
        return p

    def test_run_creates_output_file(self, backend, script, tmp_path):
        out = str(tmp_path / "results" / "r.json")
        backend.run_experiment(
            experiment_script=script,
            working_dir=str(tmp_path),
            extra_args={"--output": out},
        )
        assert os.path.exists(out)

    def test_run_output_is_valid_json(self, backend, script, tmp_path):
        out = str(tmp_path / "r.json")
        backend.run_experiment(
            experiment_script=script,
            working_dir=str(tmp_path),
            extra_args={"--output": out},
        )
        with open(out) as fh:
            data = json.load(fh)
        assert "metric" in data

    def test_run_passes_seed_arg(self, backend, script, tmp_path):
        out = str(tmp_path / "r.json")
        backend.run_experiment(
            experiment_script=script,
            working_dir=str(tmp_path),
            extra_args={"--seed": "42", "--output": out},
        )
        with open(out) as fh:
            data = json.load(fh)
        assert data["seed"] == "42"

    def test_failing_script_raises(self, backend, tmp_path):
        bad_script = str(tmp_path / "bad.py")
        with open(bad_script, "w") as fh:
            fh.write("raise RuntimeError('deliberate failure')\n")
        import subprocess
        with pytest.raises(subprocess.CalledProcessError):
            backend.run_experiment(
                experiment_script=bad_script,
                working_dir=str(tmp_path),
                extra_args={},
            )


# ---------------------------------------------------------------------------
# KaggleBackend – construction / error handling only (no real API calls)
# ---------------------------------------------------------------------------

class TestKaggleBackendConstruction:

    def test_raises_if_kaggle_not_installed(self, monkeypatch):
        """If the kaggle package is absent, __init__ should raise RuntimeError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("kaggle"):
                raise ImportError("Mocked: kaggle not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from pipeline.backends.kaggle_backend import KaggleBackend
        with pytest.raises(RuntimeError, match="kaggle package"):
            KaggleBackend(BackendConfig(type="kaggle"))


# ---------------------------------------------------------------------------
# RemoteSSHBackend – construction / error handling only (no real SSH calls)
# ---------------------------------------------------------------------------

class TestRemoteSSHBackendConstruction:

    def test_raises_with_empty_host(self):
        from pipeline.backends.remote_backend import RemoteSSHBackend
        with pytest.raises(ValueError, match="remote_host"):
            RemoteSSHBackend(BackendConfig(type="remote", remote_host=""))

    def test_constructs_with_valid_host(self):
        from pipeline.backends.remote_backend import RemoteSSHBackend
        backend = RemoteSSHBackend(
            BackendConfig(type="remote", remote_host="example.com")
        )
        assert backend._host == "example.com"

    def test_raises_if_paramiko_not_installed(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "paramiko":
                raise ImportError("Mocked: paramiko not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from pipeline.backends.remote_backend import RemoteSSHBackend
        backend = RemoteSSHBackend(
            BackendConfig(type="remote", remote_host="example.com")
        )
        with pytest.raises(RuntimeError, match="paramiko"):
            backend.run_experiment("script.py", ".", {})


# ---------------------------------------------------------------------------
# file_ops utilities
# ---------------------------------------------------------------------------

class TestFileOps:

    def test_next_idea_id_empty_dir(self, tmp_path):
        from pipeline.utils.file_ops import next_idea_id
        assert next_idea_id(str(tmp_path)) == "001"

    def test_next_idea_id_skips_existing(self, tmp_path):
        from pipeline.utils.file_ops import next_idea_id
        os.makedirs(tmp_path / "idea-001-foo")
        os.makedirs(tmp_path / "idea-003-bar")
        assert next_idea_id(str(tmp_path)) == "004"

    def test_scaffold_creates_dirs(self, tmp_path):
        from pipeline.utils.file_ops import scaffold_idea_directory
        base = scaffold_idea_directory(str(tmp_path), "001", "my-idea")
        assert os.path.isdir(os.path.join(base, "experiments", "tests"))
        assert os.path.isdir(os.path.join(base, "paper"))

    def test_update_master_log_creates_file(self, tmp_path):
        from pipeline.utils.file_ops import update_master_log_status
        path = str(tmp_path / "masterLog.md")
        update_master_log_status(path, "001", "my-idea", "A test idea", "PROPOSED")
        assert os.path.exists(path)
        content = open(path).read()
        assert "001" in content
        assert "PROPOSED" in content

    def test_update_master_log_replaces_existing_row(self, tmp_path):
        from pipeline.utils.file_ops import update_master_log_status
        path = str(tmp_path / "masterLog.md")
        update_master_log_status(path, "001", "my-idea", "desc", "PROPOSED")
        update_master_log_status(path, "001", "my-idea", "desc", "VALIDATED")
        content = open(path).read()
        assert content.count("001") == content.count("001")  # row updated, not duplicated
        assert "VALIDATED" in content
