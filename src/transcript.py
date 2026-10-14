"""Transcript storage: plain-text transcripts with legacy JSON migration.

Transcripts are stored as a single plain-text file
({lecture}.transcript.txt) — one continuous block of text with no
JSON overhead. Legacy MacWhisper JSON transcripts
({lecture}.transcript.json) are converted to txt automatically
the first time a lecture is processed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .util import atomic_write_text

TRANSCRIPT_SUFFIX = ".transcript.txt"
LEGACY_TRANSCRIPT_SUFFIX = ".transcript.json"

# A real lecture transcript is always longer than this. Shorter
# files mean the transcription failed but still produced output.
MIN_TRANSCRIPT_CHARS = 500


class TranscriptError(RuntimeError):
    """Raised when a transcript is missing, empty or unusable."""


def transcript_path(lecture_name: str, base_dir: str) -> Path:
    """Path to the plain-text transcript of a lecture."""
    return Path(base_dir) / f"{lecture_name}{TRANSCRIPT_SUFFIX}"


def legacy_transcript_path(lecture_name: str, base_dir: str) -> Path:
    """Path to the legacy JSON transcript of a lecture."""
    return Path(base_dir) / f"{lecture_name}{LEGACY_TRANSCRIPT_SUFFIX}"


def migrate_legacy_json(lecture_name: str, base_dir: str) -> Optional[Path]:
    """Convert a legacy .transcript.json into a .transcript.txt.

    Args:
        lecture_name: Lecture name (filename stem).
        base_dir: Directory containing the transcript files.

    Returns:
        Path to the txt transcript (existing or newly created),
        or None when there is no legacy JSON to convert.

    Raises:
        TranscriptError: The legacy JSON is unreadable or empty.
    """
    txt_path = transcript_path(lecture_name, base_dir)
    if txt_path.is_file():
        return txt_path

    json_path = legacy_transcript_path(lecture_name, base_dir)
    if not json_path.is_file():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise TranscriptError(
            f"Could not read legacy transcript {json_path}: {e}"
        ) from None

    parts = [
        str(seg.get("text", "")).strip()
        for seg in data.get("segments", [])
        if isinstance(seg, dict)
    ]
    text = "\n".join(part for part in parts if part)

    if len(text.strip()) < MIN_TRANSCRIPT_CHARS:
        raise TranscriptError(
            f"Legacy transcript {json_path} contains too little text "
            f"({len(text.strip())} chars)"
        )

    atomic_write_text(str(txt_path), text)
    return txt_path


def load_transcript_text(lecture_name: str, base_dir: str) -> str:
    """Load the plain-text transcript, migrating legacy JSON if needed.

    Args:
        lecture_name: Lecture name (filename stem).
        base_dir: Directory containing the transcript files.

    Returns:
        The transcript text (stripped).

    Raises:
        TranscriptError: No transcript exists or it is too short
            to be a real lecture.
    """
    txt_path = migrate_legacy_json(lecture_name, base_dir)
    if txt_path is None or not txt_path.is_file():
        raise TranscriptError(
            f"Transcript file not found: "
            f"{transcript_path(lecture_name, base_dir)}"
        )

    try:
        text = txt_path.read_text(encoding="utf-8")
    except OSError as e:
        raise TranscriptError(
            f"Could not read transcript {txt_path}: {e}"
        ) from None

    text = text.strip()
    if len(text) < MIN_TRANSCRIPT_CHARS:
        # A near-empty transcript is useless and would block
        # re-transcription on the next run — remove it.
        try:
            txt_path.unlink()
            print(f"  Removed empty transcript: {txt_path}")
        except OSError:
            pass
        raise TranscriptError(
            f"Transcript {txt_path} is too short ({len(text)} chars) — "
            f"transcription probably failed"
        )
    return text
