"""Client for the local LM Studio inference server.

LM Studio exposes an OpenAI-compatible HTTP API on localhost:1234.
This module handles automatic context sizing (estimate tokens,
add output budget and headroom), one-shot generation with retries
on transient errors, precise error reporting (context-length
failures raise ContextTooLongError), and response validation
(empty/truncated responses are detected and retried).

All configuration (model, max context, temperature) comes from the
pipeline config, not hardcoded.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

import urllib.error
import urllib.request

# Default LM Studio server URL
LM_STUDIO_DEFAULT_URL = "http://localhost:1234"

# Conservative token estimate: Cyrillic-heavy text averages roughly
# 2-3 characters per token with SentencePiece tokenizers. Using 2.0
# overestimates the real count, which is the safe direction when
# deciding whether a transcript fits the model's context window.
CHARS_PER_TOKEN = 2.0

# Safety margin on top of input + output tokens.
HEADROOM = 1.10

# Required context is rounded up to this granularity.
CONTEXT_GRANULARITY = 1024

# Completion (output) budget: fraction of input tokens, clamped.
OUTPUT_BUDGET_RATIO = 0.15
OUTPUT_BUDGET_MIN = 2048
OUTPUT_BUDGET_MAX = 16384
OUTPUT_BUDGET_HARD_CAP = 32768

# HTTP codes worth retrying (rate limits and transient server errors).
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# Substrings that mark an LM Studio error as context-related.
_CONTEXT_ERROR_MARKERS = (
    "context",
    "too long",
    "too many",
    "token",
    "length",
    "exceed",
    "maximum",
)


class LLMError(RuntimeError):
    """Base error for LLM client failures."""


class ContextTooLongError(LLMError):
    """The request does not fit into the model's context window."""


class ModelNotLoadedError(LLMError):
    """The configured model is not loaded in LM Studio."""


class EmptyResponseError(LLMError):
    """The model returned an empty response."""


class TruncatedResponseError(LLMError):
    """The model stopped generating because max_tokens was reached."""


def _looks_like_context_error(message: str) -> bool:
    """Heuristic: does an LM Studio error message concern context size?"""
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_ERROR_MARKERS)


class LLMClient:
    """Client for the LM Studio local inference API."""

    def __init__(
        self,
        base_url: str = LM_STUDIO_DEFAULT_URL,
        model: str = "",
        context_length: int = 65536,
        temperature: float = 0.2,
        request_timeout: float = 3600.0,
        max_retries: int = 3,
    ):
        """Initialize LLM client.

        Args:
            base_url: LM Studio server base URL.
            model: Model identifier to use for generation.
            context_length: Maximum context window the model may use.
            temperature: Sampling temperature for generation.
            request_timeout: Per-request socket timeout in seconds
                (generation of long notes can take minutes).
            max_retries: Retries for transient failures.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model or ""
        self.context_length = context_length
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._completions_url = f"{self.base_url}/v1/chat/completions"
        self._models_url = f"{self.base_url}/v1/models"

    # ------------------------------------------------------------------
    # Context sizing
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative token estimate for a text (overestimates).

        Args:
            text: Input text.

        Returns:
            Estimated token count, rounded up.
        """
        return int(-(-len(text) // CHARS_PER_TOKEN))

    def output_budget(self, input_tokens: int) -> int:
        """Completion budget in tokens for a given input size.

        Args:
            input_tokens: Estimated input token count.

        Returns:
            Output budget, clamped to [OUTPUT_BUDGET_MIN, OUTPUT_BUDGET_MAX].
        """
        budget = int(input_tokens * OUTPUT_BUDGET_RATIO)
        return max(OUTPUT_BUDGET_MIN, min(OUTPUT_BUDGET_MAX, budget))

    def required_context(self, prompt_text: str) -> Tuple[int, int]:
        """Compute the context window required for a prompt.

        Args:
            prompt_text: The full prompt that will be sent.

        Returns:
            (required_context, output_budget) where required_context
            includes input tokens, output budget and safety headroom,
            rounded up to CONTEXT_GRANULARITY.
        """
        input_tokens = self.estimate_tokens(prompt_text)
        budget = self.output_budget(input_tokens)
        total = int((input_tokens + budget) * HEADROOM)
        required = -(-total // CONTEXT_GRANULARITY) * CONTEXT_GRANULARITY
        return required, budget

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a chat completion.

        If the model stops because the output budget was exhausted,
        the request is retried with a doubled budget (up to
        OUTPUT_BUDGET_HARD_CAP and as far as the context window
        allows) instead of silently returning a truncated note.

        Args:
            messages: List of message dicts with "role" and "content".
            max_tokens: Output budget override.
            temperature: Temperature override for this call.

        Returns:
            The generated text content.

        Raises:
            ContextTooLongError: The request does not fit the context.
            ModelNotLoadedError: The model is not loaded.
            EmptyResponseError: The response was empty.
            LLMError: Any other failure after retries.
        """
        temp = temperature if temperature is not None else self.temperature
        if max_tokens is None:
            input_tokens = self.estimate_tokens(
                " ".join(m.get("content", "") for m in messages)
            )
            max_tokens = self.output_budget(input_tokens)

        while True:
            try:
                return self._generate_once(messages, max_tokens, temp)
            except TruncatedResponseError:
                headroom = self.context_length - max_tokens
                doubled = min(max_tokens * 2, OUTPUT_BUDGET_HARD_CAP, headroom)
                if doubled <= max_tokens:
                    raise LLMError(
                        "Model response was truncated and the output budget "
                        f"cannot be increased further (context window is "
                        f"{self.context_length} tokens)"
                    ) from None
                print(f"  Response truncated at {max_tokens} tokens, "
                      f"retrying with budget {doubled}")
                max_tokens = doubled

    def _generate_once(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Perform a single HTTP request with retries."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_length": self.context_length,
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self._completions_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=self.request_timeout
                ) as response:
                    body = response.read().decode("utf-8")
                    return self._parse_response(body)
            except urllib.error.HTTPError as e:
                error_text = self._read_error_body(e)
                if (
                    e.code in RETRYABLE_HTTP_CODES
                    and attempt < self.max_retries
                ):
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 400 and _looks_like_context_error(error_text):
                    raise ContextTooLongError(
                        f"LM Studio: {error_text}"
                    ) from None
                raise LLMError(
                    f"LM Studio HTTP {e.code}: {error_text}"
                ) from None
            except urllib.error.URLError as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(
                    f"LM Studio connection error: {e}"
                ) from None

        raise LLMError(
            f"LM Studio request failed after {self.max_retries + 1} "
            f"attempts: {last_error}"
        )

    def _parse_response(self, body: str) -> str:
        """Validate an OpenAI-style completion response."""
        try:
            result = json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse LM Studio response: {e}"
            ) from None

        choices = result.get("choices") or []
        if not choices:
            raise LLMError(
                f"No choices in LM Studio response: {body[:500]}"
            )
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") or ""
        if not content.strip():
            raise EmptyResponseError("LM Studio returned an empty response")
        if choice.get("finish_reason") == "length":
            raise TruncatedResponseError(
                "model stopped at max_tokens (response truncated)"
            )
        return content

    @staticmethod
    def _read_error_body(e: urllib.error.HTTPError) -> str:
        """Extract a human-readable message from an HTTP error response."""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            return str(e)
        try:
            parsed = json.loads(raw)
            err = parsed.get("error", parsed)
            if isinstance(err, dict):
                return str(err.get("message", raw))
            return str(err)
        except json.JSONDecodeError:
            return raw[:500]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def server_reachable(self) -> bool:
        """Check if the LM Studio server answers at all."""
        try:
            req = urllib.request.Request(
                self._models_url,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                response.read()
            return True
        except Exception:
            return False

    def is_model_loaded(self) -> bool:
        """Check if the configured model is loaded in LM Studio.

        Uses `lms ps` which lists models currently in memory.
        The /v1/models endpoint lists all available models, not
        just loaded ones, so it cannot be used for this check.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["lms", "ps"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts and parts[0] == self.model:
                    return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def load_model(self, model_key: str) -> bool:
        """Load a model into LM Studio via the `lms load` CLI."""
        import subprocess

        try:
            result = subprocess.run(
                ["lms", "load", model_key],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                self.model = model_key
                return True
            print(f"  Warning: lms load failed: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            print("  Warning: lms load timed out")
            return False
        except FileNotFoundError:
            print("  Warning: lms command not found")
            return False

    def start_server(self) -> bool:
        """Try to start the LM Studio local server.

        Runs `lms server start`, then polls for reachability.
        Falls back to opening the LM Studio app (macOS) if the
        CLI start fails.

        Returns:
            True if the server is reachable after the attempt.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["lms", "server", "start"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                print("  LM Studio server started")
            else:
                print(
                    f"  lms server start: {result.stderr.strip()[:200]}"
                )
        except subprocess.TimeoutExpired:
            print("  Warning: lms server start timed out")
        except FileNotFoundError:
            print("  Warning: lms command not found")

        # Poll for reachability (up to 60s)
        for _ in range(30):
            if self.server_reachable():
                return True
            time.sleep(2)

        # Fallback: open the LM Studio app (macOS)
        try:
            subprocess.run(
                ["open", "-a", "LM Studio"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            print("  Opened LM Studio app, waiting for server...")
            for _ in range(30):
                if self.server_reachable():
                    return True
                time.sleep(2)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    def ensure_server(self) -> None:
        """Make sure the LM Studio server is running, starting it if needed.

        Raises:
            ModelNotLoadedError: The server could not be started.
        """
        if self.server_reachable():
            return
        print("  LM Studio server not running, starting it...")
        if not self.start_server():
            raise ModelNotLoadedError(
                f"LM Studio server not reachable at {self.base_url} "
                f"— could not start it automatically"
            )

    def ensure_model_loaded(self) -> None:
        """Make sure the configured model is loaded, loading it if needed.

        Raises:
            ModelNotLoadedError: The server is unreachable or the
                model could not be loaded.
        """
        self.ensure_server()
        if self.is_model_loaded():
            return
        print(f"  Model '{self.model}' not loaded in LM Studio, loading...")
        if not self.load_model(self.model):
            raise ModelNotLoadedError(
                f"Model '{self.model}' is not loaded in LM Studio"
            )
        if not self.is_model_loaded():
            raise ModelNotLoadedError(
                f"Model '{self.model}' did not appear in LM Studio after load"
            )
