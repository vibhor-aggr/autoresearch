"""
file_ops.py – File/directory utility functions for the pipeline.

Provides helpers for:
  - scaffolding per-idea directory trees
  - reading / appending to Markdown log files
  - updating the master log table
  - writing experiment code to disk
"""

from __future__ import annotations

import os
import re
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Idea directory scaffolding
# ---------------------------------------------------------------------------

def idea_dir(ideas_root: str, idea_id: str, slug: str) -> str:
    """Return the canonical path for an idea directory."""
    return os.path.join(ideas_root, f"idea-{idea_id}-{slug}")


def scaffold_idea_directory(ideas_root: str, idea_id: str, slug: str) -> str:
    """
    Create the standard directory layout for a new idea and return its path.

    Layout::

        ideas/idea-NNN-<slug>/
            experiments/
                tests/
                results/      (gitignored)
                figures/      (gitignored)
            paper/
    """
    base = idea_dir(ideas_root, idea_id, slug)
    for sub in [
        "experiments/tests",
        "experiments/results",
        "experiments/figures",
        "paper",
    ]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def write_text(path: str, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def read_text(path: str, default: str = "") -> str:
    """Return the contents of *path*, or *default* if the file does not exist."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def append_log_entry(log_path: str, step: int, action: str, reasoning: str) -> None:
    """
    Append a chronological entry to a Markdown log file.

    Format::

        ### [STEP N] <Action>
        **Date**: YYYY-MM-DD
        **Action**: ...
        **Reasoning**: ...
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = textwrap.dedent(f"""

        ### [STEP {step}] {action}
        **Date**: {date}
        **Action**: {action}
        **Reasoning**: {reasoning}
    """)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry)


# ---------------------------------------------------------------------------
# Master log helpers
# ---------------------------------------------------------------------------

_TABLE_HEADER = (
    "| ID | Slug | One-line Description | Status | Venue Target | Last Updated |\n"
    "|---|---|---|---|---|---|\n"
)


def update_master_log_status(
    master_log_path: str,
    idea_id: str,
    slug: str,
    description: str,
    status: str,
    venue: str = "TBD",
) -> None:
    """
    Insert or update a row in the Idea Registry table of *master_log_path*.

    If a row with *idea_id* already exists it is replaced; otherwise a new
    row is appended.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_row = f"| {idea_id} | {slug} | {description} | `{status}` | {venue} | {date} |\n"

    content = read_text(master_log_path, _default_master_log())

    # Try to replace an existing row
    pattern = re.compile(rf"^\|\s*{re.escape(idea_id)}\s*\|.*$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(new_row.rstrip("\n"), content)
    else:
        # Append after the last table row
        if _TABLE_HEADER.split("\n")[0] in content:
            # Find the table block and append
            lines = content.splitlines(keepends=True)
            # Find last row of the table
            in_table = False
            insert_pos = len(lines)
            for i, line in enumerate(lines):
                if "| ID |" in line:
                    in_table = True
                if in_table and line.startswith("|"):
                    insert_pos = i + 1
                elif in_table and not line.startswith("|"):
                    break
            lines.insert(insert_pos, new_row)
            content = "".join(lines)
        else:
            content += f"\n\n## Idea Registry\n\n{_TABLE_HEADER}{new_row}"

    # Append a change-log entry
    change_entry = (
        f"| {date} | Updated idea {idea_id} ({slug}) status → {status} |\n"
    )
    if "## Change Log" in content:
        content = content.rstrip() + "\n" + change_entry
    else:
        content += f"\n\n## Change Log\n\n| Date | Entry |\n|---|---|\n{change_entry}"

    write_text(master_log_path, content)


def _default_master_log() -> str:
    return (
        "# Master Research Log\n\n"
        "This file is the top-level index for all autonomous research activity.\n\n"
        "---\n\n"
        "## Idea Registry\n\n"
        + _TABLE_HEADER
    )


# ---------------------------------------------------------------------------
# Experiment code helpers
# ---------------------------------------------------------------------------

def write_experiment_code(
    experiments_dir: str,
    filename: str,
    code: str,
) -> str:
    """Write *code* to ``<experiments_dir>/<filename>`` and return the full path."""
    path = os.path.join(experiments_dir, filename)
    write_text(path, code)
    return path


def write_requirements(experiments_dir: str, packages: list) -> str:
    """Write a ``requirements.txt`` from a list of package specs."""
    path = os.path.join(experiments_dir, "requirements.txt")
    content = "\n".join(packages) + "\n"
    write_text(path, content)
    return path


# ---------------------------------------------------------------------------
# Next idea ID helper
# ---------------------------------------------------------------------------

def next_idea_id(ideas_root: str) -> str:
    """
    Return the next zero-padded three-digit idea ID by scanning *ideas_root*.

    Examples: ``"001"``, ``"003"``, ``"010"``.
    """
    if not os.path.exists(ideas_root):
        return "001"
    ids = []
    for name in os.listdir(ideas_root):
        m = re.match(r"idea-(\d+)-", name)
        if m:
            ids.append(int(m.group(1)))
    return f"{max(ids, default=0) + 1:03d}"


# ---------------------------------------------------------------------------
# Existing ideas summary (for de-duplication prompting)
# ---------------------------------------------------------------------------

def existing_ideas_summary(ideas_root: str) -> str:
    """
    Return a concise summary of all existing ideas for inclusion in LLM prompts
    so the model can avoid proposing duplicates.
    """
    if not os.path.exists(ideas_root):
        return "No existing ideas yet."
    lines = []
    for name in sorted(os.listdir(ideas_root)):
        m = re.match(r"idea-(\d+)-(.+)", name)
        if not m:
            continue
        idea_id, slug = m.group(1), m.group(2)
        spec_path = os.path.join(ideas_root, name, "specification.md")
        if os.path.exists(spec_path):
            with open(spec_path, encoding="utf-8") as fh:
                first_line = fh.readline().strip()
        else:
            first_line = slug
        lines.append(f"  - [{idea_id}] {first_line}")
    return "\n".join(lines) if lines else "No existing ideas yet."
