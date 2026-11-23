"""Disk-use policy.

Nothing is written to disk unless the caller names an output directory, and
that directory may not be on the C: drive. Inputs fetched from SEC and Yahoo
stay in memory; workbooks and CSVs are built as in-memory bytes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

ENV_DIR = "VALUATION_OUTPUT_DIR"
_C_DRIVE = re.compile(r"^[cC]:")


class StorageRefused(ValueError):
    pass


def _drive_of(path: Path) -> str:
    text = str(path)
    if os.name == "nt":
        return PureWindowsPath(text).drive
    # WSL / Git-Bash style mounts of the Windows C: drive.
    lowered = text.replace("\\", "/").lower()
    if lowered.startswith("/mnt/c/") or lowered == "/mnt/c" or lowered.startswith("/c/"):
        return "C:"
    return PureWindowsPath(text).drive


def is_c_drive(path: str | Path) -> bool:
    """True for C:\\..., c:/..., /mnt/c/..., and relative paths that resolve onto C: (e.g. a repo on C:)."""
    text = str(path).strip()
    lowered = text.replace("\\", "/").lower()
    if _C_DRIVE.match(text) or lowered.startswith(("/mnt/c/", "/c/")) or lowered in ("/mnt/c", "/c"):
        return True
    if PureWindowsPath(text).drive:  # another drive letter, e.g. D:
        return False
    resolved = Path(text).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    return bool(_C_DRIVE.match(_drive_of(resolved)))


def output_dir(requested: str | Path | None) -> Path | None:
    """The directory files may be written to, or None for memory-only operation."""
    chosen = requested or os.environ.get(ENV_DIR) or None
    if not chosen:
        return None
    path = Path(chosen).expanduser()
    if is_c_drive(path):
        raise StorageRefused(
            f"Refusing to write to {path}: it is on the C: drive. Choose a folder on another drive "
            f"(for example D:\\valuation_out) or omit the output folder to keep everything in memory."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path
