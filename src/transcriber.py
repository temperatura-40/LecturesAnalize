"""Transcription via the MacWhisper CLI (mw).

Runs `mw transcribe` and stores the result as a plain-text file
({lecture}.transcript.txt) next to the audio file. Plain text is
all the pipeline needs — no JSON with timestamps and word-level
detail that inflates the file size ~25x.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .transcript import transcript_path


class TranscriptionError(RuntimeError):
    """Raised when MacWhisper fails to produce a transcript."""


class MacWhisperTranscriber:
    """Thin wrapper around the `mw transcribe` CLI command."""

    def __init__(
        self,
        model: str = "parakeet-pro:nvidia_parakeet-v3",
        language: str = "ru",
    ):
        """Initialize the transcriber.

        Args:
            model: MacWhisper model (engine:model-id format).
            language: Source language (ISO 639-1 code).
        """
        self.model = model
        self.language = language

    def transcribe(self, lecture_name: str, audio_path: str) -> Path:
        """Transcribe an audio file to a plain-text transcript.

        Args:
            lecture_name: Lecture name (filename stem).
            audio_path: Path to the audio/video file.

        Returns:
            Path to the created .transcript.txt file.

        Raises:
            TranscriptionError: Audio missing, mw failed, or the
                produced transcript is empty.
        """
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            raise TranscriptionError(
                f"Audio file not found: {audio_path}"
            )

        out_path = transcript_path(lecture_name, str(audio_file.parent))

        cmd = [
            "mw",
            "transcribe",
            "--format",
            "txt",
            "--no-timestamps",
            "--language",
            self.language,
            "--model",
            self.model,
            "--overwrite",
            "--output",
            str(out_path),
            str(audio_file),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            raise TranscriptionError(
                "MacWhisper timed out after 1 hour"
            ) from None
        except FileNotFoundError:
            raise TranscriptionError(
                "MacWhisper CLI (mw) not found in PATH"
            ) from None

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise TranscriptionError(
                f"MacWhisper failed (exit {result.returncode}): "
                f"{stderr[:500]}"
            )

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise TranscriptionError(
                "MacWhisper finished but produced an empty transcript"
            )

        return out_path
