#!/usr/bin/env python3
"""CLI entry point for the lecture notes pipeline.

Main command:
    python cli.py process   - Process all lectures (transcript -> note)

Auxiliary commands:
    python cli.py status    - Show status of all lectures
    python cli.py transcribe - Transcribe lectures (skip LLM)
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import Pipeline


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python cli.py <command>")
        print()
        print("Commands:")
        print("  process   - Process all lectures through the pipeline")
        print("  status    - Show status of all lectures")
        print("  transcribe - Transcribe lectures (skip the LLM step)")
        sys.exit(1)

    command = sys.argv[1].lower()

    # Optional: path to a config file
    config_path = None
    if "--config" in sys.argv[2:]:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]

    # Initialize pipeline
    pipeline = Pipeline(config_path)

    if command == "process":
        failures = pipeline.process()
        sys.exit(1 if failures else 0)
    elif command == "status":
        pipeline.status()
    elif command == "transcribe":
        failures = pipeline.transcribe_only()
        sys.exit(1 if failures else 0)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: process, status, transcribe")
        sys.exit(1)


if __name__ == "__main__":
    main()
