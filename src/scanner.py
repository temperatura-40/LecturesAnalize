"""Scanner for discovering lecture audio files and their artifacts.

Finds audio files in the audio folder supported by MacWhisper.
Artifacts are matched in the audio file's own directory only:
  {name}.transcript.txt   - plain-text transcript (current format)
  {name}.transcript.json  - legacy JSON transcript (auto-converted)
  {name}.md               - generated lecture note

This keeps discovery predictable: an audio file and its
artifacts live together in one folder. Moving artifacts
out of that folder (for example into an archive subfolder)
makes the pipeline process the lecture from scratch again,
including transcription.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Supported audio extensions (lowercase)
SUPPORTED_EXTENSIONS: tuple = (
    ".m4a",
    ".mp3",
    ".wav",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
)


def scan_lectures(audio_folder: str) -> Dict[str, Dict[str, object]]:
    """Scan for lecture audio files and their artifacts.

    Walks the audio folder looking for audio files. For each
    audio file found, checks its own directory for existing
    artifacts (transcript, legacy JSON transcript, note).

    Args:
        audio_folder: Path to folder containing audio files.

    Returns:
        Dictionary keyed by audio filename stem with values:
        {
            "audio_path": str,              // Full path to audio file
            "relative_name": str,             // Filename without extension
            "has_transcript": bool,         // .transcript.txt next to audio
            "has_legacy_transcript": bool,  // .transcript.json next to audio
            "has_md": bool,                   // .md next to audio
        }
    """
    audio_path = Path(audio_folder).expanduser().resolve()

    audio_files: Dict[str, Dict[str, object]] = {}
    if not audio_path.is_dir():
        return audio_files

    for root, _dirs, files in os.walk(audio_path):
        for f in files:
            if Path(f).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_path = Path(root) / f
            rel_name = Path(f).stem
            parent = file_path.parent
            audio_files[rel_name] = {
                "audio_path": str(file_path.resolve()),
                "relative_name": rel_name,
                "has_transcript": (
                    parent / f"{rel_name}.transcript.txt"
                ).is_file(),
                "has_legacy_transcript": (
                    parent / f"{rel_name}.transcript.json"
                ).is_file(),
                "has_md": (parent / f"{rel_name}.md").is_file(),
            }

    return audio_files
