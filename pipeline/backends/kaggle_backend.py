"""
kaggle_backend.py – Kaggle Notebooks backend.

Pushes the experiment code as a Kaggle Dataset, creates a Kernel (notebook)
that runs it, polls for completion, and downloads the output.

Prerequisites
-------------
``pip install kaggle``

Environment variables required:
    KAGGLE_USERNAME   – your Kaggle username
    KAGGLE_KEY        – your Kaggle API key   (or set KAGGLE_KEY env var)

Alternatively, set ``~/.kaggle/kaggle.json`` as per the Kaggle API docs.

Notes
-----
- Kaggle Kernels run for a maximum of ~9 hours (GPU).
- Output files must be saved to ``/kaggle/working/`` inside the kernel.
- This backend writes the output JSON to the local ``results/`` directory
  after downloading the kernel output.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Dict

from pipeline.backends.base import AbstractBackend
from pipeline.config import BackendConfig


class KaggleBackend(AbstractBackend):
    """
    Execute experiment scripts via the Kaggle Kernels API.

    Falls back gracefully if the ``kaggle`` package is not installed or
    credentials are missing – in that case it raises :exc:`RuntimeError`
    with actionable instructions.
    """

    _POLL_INTERVAL = 30  # seconds between status checks

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._username = (
            config.kaggle_username
            or os.environ.get("KAGGLE_USERNAME", "")
        )
        self._key = os.environ.get(config.kaggle_key_env, "")
        self._api = self._init_api()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_api(self):
        try:
            import kaggle  # type: ignore  # noqa: F401
            from kaggle.api.kaggle_api_extended import KaggleApiExtended  # type: ignore
            api = KaggleApiExtended()
            api.authenticate()
            return api
        except ImportError as exc:
            raise RuntimeError(
                "kaggle package not found. Install with: pip install kaggle\n"
                "Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Kaggle authentication failed: {exc}\n"
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY are set, or create "
                "~/.kaggle/kaggle.json."
            ) from exc

    # ------------------------------------------------------------------
    # AbstractBackend implementation
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        experiment_script: str,
        working_dir: str,
        extra_args: Dict[str, str],
    ) -> None:
        """
        1. Bundle the experiment script into a temporary directory.
        2. Push it as a Kaggle Dataset.
        3. Create / update a Kernel that calls the script.
        4. Wait for completion.
        5. Download results and copy to ``extra_args["--output"]``.
        """
        output_path = extra_args.get("--output", "results.json")
        seed        = extra_args.get("--seed", "0")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy experiment script
            shutil.copy(experiment_script, tmpdir)
            script_name = os.path.basename(experiment_script)

            # Write a wrapper notebook script
            notebook_code = self._build_notebook_code(
                script_name, seed, output_path
            )
            notebook_path = os.path.join(tmpdir, "notebook.py")
            with open(notebook_path, "w") as fh:
                fh.write(notebook_code)

            # Create dataset metadata
            dataset_slug = f"autoresearch-exp-seed{seed}"
            meta = {
                "title": dataset_slug,
                "id": f"{self._username}/{dataset_slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }
            with open(os.path.join(tmpdir, "dataset-metadata.json"), "w") as fh:
                json.dump(meta, fh)

            # Push dataset (creates or updates)
            try:
                self._api.dataset_create_new(
                    folder=tmpdir,
                    public=False,
                    quiet=True,
                )
            except Exception:
                # Dataset may already exist; update instead
                self._api.dataset_create_version(
                    folder=tmpdir,
                    version_notes="auto-update",
                    quiet=True,
                )

            # Create / run kernel
            kernel_slug = f"autoresearch-run-seed{seed}"
            kernel_meta = {
                "id": f"{self._username}/{kernel_slug}",
                "title": kernel_slug,
                "code_file": "notebook.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "dataset_sources": [f"{self._username}/{dataset_slug}"],
            }
            kernel_meta_path = os.path.join(tmpdir, "kernel-metadata.json")
            with open(kernel_meta_path, "w") as fh:
                json.dump(kernel_meta, fh)

            self._api.kernels_push(tmpdir)

            # Wait for completion
            self._wait_for_kernel(kernel_slug)

            # Download output
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(out_dir, exist_ok=True)
            self._api.kernels_output(
                f"{self._username}/{kernel_slug}",
                path=out_dir,
                quiet=True,
            )

            # Copy result file to expected output path
            result_candidate = os.path.join(out_dir, "results.json")
            if os.path.exists(result_candidate):
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                shutil.copy(result_candidate, output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_kernel(self, kernel_slug: str) -> None:
        """Poll until the kernel finishes or times out."""
        timeout = self.config.remote_timeout  # reuse the remote timeout field
        elapsed = 0
        while elapsed < timeout:
            status_obj = self._api.kernel_status(
                f"{self._username}/{kernel_slug}"
            )
            status = getattr(status_obj, "status", "unknown")
            if status in ("complete", "error", "cancelled"):
                if status != "complete":
                    raise RuntimeError(
                        f"Kaggle kernel {kernel_slug!r} finished with status '{status}'."
                    )
                return
            time.sleep(self._POLL_INTERVAL)
            elapsed += self._POLL_INTERVAL
        raise TimeoutError(
            f"Kaggle kernel {kernel_slug!r} did not finish within {timeout}s."
        )

    @staticmethod
    def _build_notebook_code(script_name: str, seed: str, output_path: str) -> str:
        return (
            f"import subprocess, shutil, os\n"
            f"subprocess.run(\n"
            f"    ['python', '/kaggle/input/autoresearch-exp-seed{seed}/{script_name}',\n"
            f"     '--seed', '{seed}', '--output', '/kaggle/working/results.json'],\n"
            f"    check=True\n"
            f")\n"
        )
