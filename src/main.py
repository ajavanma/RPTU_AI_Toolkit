"""Compatibility entry point; prefer the installed ``rptu`` command."""

from rptu_ai_toolkit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
