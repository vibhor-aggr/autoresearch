"""
remote_backend.py – Remote SSH backend via Paramiko.

Executes experiment scripts on a remote server over SSH / SFTP.

Prerequisites
-------------
``pip install paramiko``

Configuration fields used from :class:`BackendConfig`:
    remote_host       – hostname or IP
    remote_user       – SSH username (default: ``ubuntu``)
    remote_key_file   – path to private SSH key (default: ``~/.ssh/id_rsa``)
    remote_work_dir   – working directory on remote server
    remote_timeout    – max seconds to wait for the experiment to finish

The experiment script and its requirements are copied to *remote_work_dir*
via SFTP.  After execution, the results JSON is fetched back locally.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict

from pipeline.backends.base import AbstractBackend
from pipeline.config import BackendConfig


class RemoteSSHBackend(AbstractBackend):
    """
    Execute experiment scripts on a remote server via SSH.

    Raises :exc:`RuntimeError` if Paramiko is not installed or SSH
    authentication fails.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._host     = config.remote_host
        self._user     = config.remote_user
        self._key_file = os.path.expanduser(config.remote_key_file)
        self._work_dir = config.remote_work_dir
        self._timeout  = config.remote_timeout
        self._known_hosts = os.path.expanduser(
            getattr(config, "remote_known_hosts", "~/.ssh/known_hosts")
        )

        if not self._host:
            raise ValueError(
                "remote_host must be set in BackendConfig when using the "
                "remote backend."
            )

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
        1. Open SSH + SFTP connection to the remote host.
        2. Ensure the remote working directory exists.
        3. Upload the experiment script.
        4. Run the script remotely with the given args.
        5. Download the output JSON back to the local ``--output`` path.
        """
        ssh = self._connect()
        try:
            sftp = ssh.open_sftp()
            try:
                # Ensure remote work dir exists
                _remote_makedirs(ssh, self._work_dir)

                # Upload script
                script_name = os.path.basename(experiment_script)
                remote_script = f"{self._work_dir}/{script_name}"
                sftp.put(experiment_script, remote_script)

                # Build remote command
                output_path = extra_args.get("--output", "results.json")
                remote_output = f"{self._work_dir}/results.json"
                args_str = " ".join(
                    f"{k} {v}" for k, v in extra_args.items()
                    if k != "--output"
                )
                cmd = (
                    f"cd {self._work_dir} && "
                    f"python {remote_script} {args_str} "
                    f"--output {remote_output}"
                )

                # Execute
                _, stdout, stderr = ssh.exec_command(cmd, timeout=self._timeout)
                exit_code = stdout.channel.recv_exit_status()
                if exit_code != 0:
                    err = stderr.read().decode(errors="replace")
                    raise RuntimeError(
                        f"Remote experiment failed (exit {exit_code}): {err}"
                    )

                # Download results
                os.makedirs(
                    os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True,
                )
                sftp.get(remote_output, output_path)
            finally:
                sftp.close()
        finally:
            ssh.close()

    # ------------------------------------------------------------------
    # SSH connection helper
    # ------------------------------------------------------------------

    def _connect(self):
        try:
            import paramiko  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paramiko package is required for the remote SSH backend. "
                "Install with: pip install paramiko"
            ) from exc

        client = paramiko.SSHClient()
        # Load system-wide and user known_hosts for host key verification.
        client.load_system_host_keys()
        if self._known_hosts and os.path.exists(self._known_hosts):
            client.load_host_keys(self._known_hosts)
        # RejectPolicy: refuse connections to hosts not in known_hosts.
        # This prevents man-in-the-middle attacks.
        # To add a new host, run: ssh-keyscan <host> >> ~/.ssh/known_hosts
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(
                hostname=self._host,
                username=self._user,
                key_filename=self._key_file,
                timeout=30,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SSH connection to {self._user}@{self._host} failed: {exc}\n"
                "If connecting to a new host, add it to known_hosts first:\n"
                f"  ssh-keyscan {self._host} >> ~/.ssh/known_hosts"
            ) from exc
        return client


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _remote_makedirs(ssh, path: str) -> None:
    """Create *path* and all parents on the remote host."""
    ssh.exec_command(f"mkdir -p {path}")
