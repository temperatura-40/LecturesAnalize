"""Small file helpers: atomic writes and copies.

Atomic writes (temp file + rename) prevent corrupt/partial documents
if the process crashes or the disk fills up mid-write. A failed write
never leaves a half-written .md or .txt behind.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_text(path: str, content: str) -> None:
    """Write text to path atomically (temp file + rename).

    Args:
        path: Destination file path.
        content: Full text to write.

    Raises:
        OSError: If the write fails (temp file is cleaned up).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_copy(src: str, dst: str) -> None:
    """Copy a file atomically (temp copy in destination dir + rename).

    Args:
        src: Source file path.
        dst: Destination file path.

    Raises:
        OSError: If the copy fails (temp file is cleaned up).
    """
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(dst_path.parent), suffix=".tmp")
    try:
        os.close(fd)
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
