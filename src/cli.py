#!/usr/bin/env python3
"""Interactive CLI for the lecture notes pipeline.

Usage:
    lectures              Launch interactive menu
    lectures process      Process all lectures (non-interactive)
    lectures status       Show status (non-interactive)
    lectures transcribe   Transcribe only (non-interactive)
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import Pipeline


def run_interactive():
    """Launch interactive menu with curses."""
    import curses

    def draw_menu(stdscr, selected_idx, options):
        """Draw the menu on screen."""
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()

        # Title
        title = "📚 Lecture Notes Pipeline"
        stdscr.addstr(1, 2, title, curses.A_BOLD | curses.color_pair(1))

        # Subtitle
        stdscr.addstr(2, 2, "Use ↑↓ to navigate, Enter to select", curses.A_DIM)

        # Menu options
        for i, (label, _) in enumerate(options):
            y = 4 + i
            if i == selected_idx:
                stdscr.addstr(y, 4, f"▶ {label}", curses.A_REVERSE | curses.color_pair(2))
            else:
                stdscr.addstr(y, 4, f"  {label}")

        stdscr.refresh()

    def main_loop(stdscr):
        """Main interactive loop."""
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)

        options = [
            ("Process all lectures", "process"),
            ("Transcribe only", "transcribe"),
            ("Show status", "status"),
            ("Exit", "exit"),
        ]

        selected = 0
        while True:
            draw_menu(stdscr, selected, options)
            key = stdscr.getch()

            if key == curses.KEY_UP:
                selected = (selected - 1) % len(options)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(options)
            elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                action = options[selected][1]
                if action == "exit":
                    return
                # Run the selected action
                stdscr.clear()
                stdscr.addstr(1, 2, f"Running: {options[selected][0]}...", curses.A_BOLD)
                stdscr.refresh()
                return action

    # Run curses and get selected action
    action = curses.wrapper(main_loop)

    if action == "process":
        pipeline = Pipeline()
        failures = pipeline.process()
        input("\nPress Enter to return to menu...")
        return 1 if failures else 0
    elif action == "transcribe":
        pipeline = Pipeline()
        failures = pipeline.transcribe_only()
        input("\nPress Enter to return to menu...")
        return 1 if failures else 0
    elif action == "status":
        pipeline = Pipeline()
        pipeline.status()
        input("\nPress Enter to return to menu...")
        return 0
    return 0


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        # No arguments — launch interactive menu
        sys.exit(run_interactive())

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
