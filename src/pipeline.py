"""Main pipeline: audio -> transcript -> single LLM pass -> Markdown note.

The whole transcript is sent to the model in one request, so there
are no chunks, no intermediate JSON analysis files and no synthesis
step. The required context window is computed automatically from
the transcript length after transcription; if it does not fit the
model, the lecture is skipped with a clear error and no document
is written.

Failure guarantees:
- A lecture is only written to disk after the model returned a
  validated, non-truncated Markdown note.
- Writes are atomic (temp file + rename), so a crash never leaves
  a partial document behind.
- An existing good .md is never overwritten by a failed run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.llm import (
    ContextTooLongError,
    LLMClient,
    LLMError,
    ModelNotLoadedError,
)
from src.scanner import scan_lectures
from src.transcriber import MacWhisperTranscriber, TranscriptionError
from src.transcript import (
    TranscriptError,
    load_transcript_text,
    migrate_legacy_json,
    transcript_path,
)
from src.util import atomic_copy, atomic_write_text

# Prompt templates live in <project root>/prompts
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# A valid lecture note is at least this long and contains at least
# one Markdown heading or list item.
MIN_MARKDOWN_CHARS = 200


class Pipeline:
    """Process lecture recordings into Markdown notes."""

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)

        self.folders = self.cfg.get("folders", {})
        self.macwhisper = self.cfg.get("macwhisper", {})
        self.llm_config = self.cfg.get("llm", {})

        self.transcriber = MacWhisperTranscriber(
            model=self.macwhisper.get(
                "model", "parakeet-pro:nvidia_parakeet-v3"
            ),
            language=self.macwhisper.get("language", "ru"),
        )

        self.llm = LLMClient(
            base_url=self.llm_config.get(
                "base_url", "http://localhost:1234"
            ),
            model=self.llm_config.get("model", "google/gemma-4-e4b"),
            context_length=self.llm_config.get("max_context", 65536),
            temperature=self.llm_config.get("temperature", 0.2),
        )

        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the lecture-note prompt template."""
        prompt_path = PROMPTS_DIR / "lecture.md"
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise RuntimeError(
                f"Could not load prompt file {prompt_path}: {e}"
            ) from None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_lectures(self) -> Dict[str, Dict[str, Any]]:
        """Discover lectures from the configured audio folder."""
        audio_folder = self.folders.get("audio", "~/Lectures/Audio")
        obsidian_folder = self.folders.get("obsidian", "~/Obsidian/Lectures")

        obsidian_path = Path(obsidian_folder).expanduser().resolve()
        obsidian_path.mkdir(parents=True, exist_ok=True)

        return scan_lectures(str(audio_folder))

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def process(self) -> int:
        """Run the full pipeline on all discovered lectures.

        Returns:
            Number of lectures that failed (0 = all good).
        """
        print("DISCOVER: Scanning for lectures...")
        lectures = self._discover_lectures()
        print(f"Found {len(lectures)} lecture(s)\n")

        if not lectures:
            print("No lectures found. Exiting.")
            return 0

        # Make sure LM Studio is running before processing
        try:
            self.llm.ensure_server()
        except ModelNotLoadedError as e:
            print(f"ERROR: {e}")
            return len(lectures)

        failures: List[str] = []
        for lecture_name, lecture_info in sorted(lectures.items()):
            try:
                ok = self._process_lecture(lecture_name, lecture_info)
            except Exception as e:  # noqa: BLE001 - last-resort guard
                print(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
                ok = False
            if not ok:
                failures.append(lecture_name)

        print("\n" + "=" * 50)
        total = len(lectures)
        if failures:
            print(
                f"Done with errors: {total - len(failures)}/{total} "
                f"succeeded, {len(failures)} failed:"
            )
            for name in failures:
                print(f"  FAILED: {name}")
        else:
            print(f"All {total} lecture(s) processed successfully.")
        return len(failures)

    def status(self) -> None:
        """Show status of all discovered lectures."""
        print("STATUS: Scanning for lectures...")
        lectures = self._discover_lectures()

        if not lectures:
            print("No lectures found.")
            return

        for lecture_name, info in sorted(lectures.items()):
            print(f"\n[{lecture_name}]")
            print(f"  Audio: {info['audio_path']}")

            if info["has_transcript"]:
                t_status = "yes"
            elif info["has_legacy_transcript"]:
                t_status = "legacy JSON (will convert on next run)"
            else:
                t_status = "no"
            print(f"  Transcript: {t_status}")

            txt = transcript_path(
                lecture_name, os.path.dirname(str(info["audio_path"]))
            )
            if txt.is_file():
                try:
                    text = txt.read_text(encoding="utf-8")
                    tokens = self.llm.estimate_tokens(text)
                    print(
                        f"  Length: {len(text):,} chars "
                        f"(~{tokens:,} tokens est.)"
                    )
                except OSError:
                    pass

            print(f"  Markdown: {'yes' if info['has_md'] else 'no'}")

    def transcribe_only(self) -> int:
        """Transcribe all lectures without analysis/synthesis.

        Returns:
            Number of failed transcriptions.
        """
        print("TRANSCRIBE: Scanning for lectures...")
        lectures = self._discover_lectures()

        if not lectures:
            print("No lectures found. Exiting.")
            return 0

        failures: List[str] = []
        for lecture_name, info in sorted(lectures.items()):
            if info["has_transcript"]:
                print(f"[{lecture_name}] Transcript: already exists")
                continue
            print(f"[{lecture_name}] Transcribing...")
            try:
                self._ensure_transcript(lecture_name, info)
            except (TranscriptionError, TranscriptError) as e:
                print(f"  ERROR: {e}")
                failures.append(lecture_name)

        print("\nTranscription complete.")
        return len(failures)

    # ------------------------------------------------------------------
    # Per-lecture processing
    # ------------------------------------------------------------------

    def _process_lecture(self, lecture_name: str, info: Dict[str, Any]) -> bool:
        """Process a single lecture. Returns True on success."""
        audio_path = str(info["audio_path"])
        base_dir = os.path.dirname(audio_path)

        print(f"[{lecture_name}] Processing...")

        # 1. Transcript: transcribe, migrate legacy JSON, or reuse txt
        try:
            self._ensure_transcript(lecture_name, info)
            transcript_text = load_transcript_text(lecture_name, base_dir)
        except (TranscriptionError, TranscriptError) as e:
            print(f"  ERROR: {e}")
            return False

        print(f"  Transcript: {len(transcript_text):,} chars")

        # 2. Markdown already exists -> nothing to generate,
        #    just make sure it is exported to Obsidian.
        if info.get("has_md"):
            print("  Markdown: already exists, skipping LLM")
            self._export_to_obsidian(lecture_name, base_dir)
            print(f"  [{lecture_name}] Done (skipped)\n")
            return True

        # 3. Automatic context sizing from the transcript length
        prompt_text = (
            f"{self.system_prompt}\n\n"
            f"--- TRANSCRIPT START ---\n"
            f"{transcript_text}\n"
            f"--- TRANSCRIPT END ---"
        )
        required_context, output_budget = self.llm.required_context(
            prompt_text
        )
        est_input = self.llm.estimate_tokens(prompt_text)
        print(
            f"  Estimated: ~{est_input:,} input tokens, "
            f"{output_budget:,} output budget"
        )
        print(
            f"  Required context: {required_context:,} tokens "
            f"(model max: {self.llm.context_length:,})"
        )

        if required_context > self.llm.context_length:
            print(
                f"  ERROR: Transcript too long for the model context "
                f"(needs ~{required_context:,} tokens, model supports "
                f"{self.llm.context_length:,}). Skipping — no file written."
            )
            return False

        # 4. Single LLM pass over the whole transcript
        try:
            self.llm.ensure_model_loaded()
            markdown = self.llm.generate(
                messages=[{"role": "user", "content": prompt_text}]
            )
        except (ContextTooLongError, ModelNotLoadedError, LLMError) as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping {lecture_name} — no file written.")
            return False

        # 5. Validate the generated note before touching the disk
        markdown = self._clean_markdown(markdown)
        try:
            self._validate_markdown(markdown)
        except ValueError as e:
            print(f"  ERROR: Model returned an unusable note: {e}")
            print(f"  Skipping {lecture_name} — no file written.")
            return False

        # 6. Save atomically and export to Obsidian
        md_path = Path(base_dir) / f"{lecture_name}.md"
        atomic_write_text(str(md_path), markdown)
        print(f"  -> Markdown saved to {md_path}")

        self._export_to_obsidian(lecture_name, base_dir)

        print(f"  [{lecture_name}] Done!\n")
        return True

    def _ensure_transcript(
        self, lecture_name: str, info: Dict[str, Any]
    ) -> None:
        """Make sure a plain-text transcript exists.

        Creates it by transcription, or by converting a legacy
        JSON transcript. Raises on failure.
        """
        base_dir = os.path.dirname(str(info["audio_path"]))

        if info.get("has_transcript"):
            print("  Transcript: exists")
            return

        if info.get("has_legacy_transcript"):
            print("  Transcript: legacy JSON found, converting to txt...")
            migrate_legacy_json(lecture_name, base_dir)
            print("  -> Converted")
            return

        print("  Transcript: not found, running MacWhisper...")
        out_path = self.transcriber.transcribe(
            lecture_name, str(info["audio_path"])
        )
        print(f"  -> Saved to {out_path}")

    # ------------------------------------------------------------------
    # Output validation and export
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Clean up the model's output before saving.

        Removes wrapping code fences, trailing horizontal rules and
        trailing italic meta-notes the model sometimes appends
        ("*(Примечание: ...)*"). Legitimate content is untouched.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].strip() == "":
                lines = lines[:-1]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines)

        lines = stripped.splitlines()
        while lines:
            last = lines[-1].strip()
            if not last:
                lines = lines[:-1]
                continue
            is_italic_para = last.startswith("*") and last.endswith("*")
            is_hr = last in ("***", "---", "___")
            if is_italic_para or is_hr:
                lines = lines[:-1]
                continue
            break
        return "\n".join(lines)

    @staticmethod
    def _validate_markdown(text: str) -> None:
        """Check that a generated note is a real Markdown document.

        Raises:
            ValueError: The note is too short or has no Markdown
                structure (no headings or list items).
        """
        cleaned = text.strip()
        if len(cleaned) < MIN_MARKDOWN_CHARS:
            raise ValueError(f"too short ({len(cleaned)} chars)")
        has_structure = any(
            line.lstrip().startswith(("#", "-", "*", "+", "1."))
            for line in cleaned.splitlines()
        )
        if not has_structure:
            raise ValueError(
                "no Markdown structure (no headings or lists found)"
            )

    def _export_to_obsidian(self, lecture_name: str, base_dir: str) -> None:
        """Copy the generated .md file to the Obsidian folder."""
        md_path = Path(base_dir) / f"{lecture_name}.md"
        if not md_path.is_file():
            print(f"  WARNING: {md_path} not found, cannot export")
            return

        obsidian_folder = self.folders.get("obsidian", "~/Obsidian/Lectures")
        obsidian_path = Path(obsidian_folder).expanduser().resolve()
        obsidian_path.mkdir(parents=True, exist_ok=True)

        dest_path = obsidian_path / md_path.name
        try:
            atomic_copy(str(md_path), str(dest_path))
            print(f"  -> Copied to Obsidian: {dest_path}")
        except Exception as e:  # noqa: BLE001 - export is best-effort
            print(f"  -> Error copying to Obsidian: {e}")
