"""Noninteractive commands that do not initialize an ML runtime."""

import argparse
from importlib import metadata
import json
import platform
import sys
from collections.abc import Sequence

from . import __version__


PLANNED_COMMANDS = ("inspect", "preprocess", "train", "evaluate", "predict", "benchmark")


def _package_version(name: str) -> str | None:
    """Inspect distribution metadata without importing optional dependencies."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _doctor(as_json: bool) -> int:
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("PyYAML", "torch", "warpconvnet")
        },
        "capabilities": {
            "config_validation": True,
            "ml_pipeline": False,
            "gpu_runtime": "not_checked",
        },
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"RPTU AI Toolkit {report['version']}")
        print(f"Python: {report['python']}")
        print(f"Platform: {report['platform']}")
        for name, version in report["packages"].items():
            print(f"{name}: {version or 'not installed'}")
        print("Configuration validation: available")
        print("ML pipeline: not implemented")
        print("GPU runtime: not checked (package presence does not validate CUDA)")
    return 0


def _validate_config(path: str, as_json: bool) -> int:
    # Help, version, and diagnostics also work before any YAML is loaded.
    from .config import ConfigError, load_config

    try:
        config = load_config(path)
    except ConfigError as error:
        print(f"rptu: error: {error}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print(f"Configuration is valid: {path}")
        print(f"Input directory: {config.paths.input_dir}")
        print(f"Output directory: {config.paths.output_dir}")
        print("Dataset paths and GPU availability have not been checked.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rptu",
        description="RPTU AI Toolkit: configuration and CLI foundation.",
        epilog="ML pipeline commands are planned and are not implemented in this version.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    doctor = commands.add_parser(
        "doctor", help="report the environment without loading ML libraries", allow_abbrev=False
    )
    doctor.add_argument("--json", action="store_true", help="print machine-readable diagnostics")

    config = commands.add_parser(
        "config", help="validate application configuration", allow_abbrev=False
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser(
        "validate", help="check configuration without running a pipeline", allow_abbrev=False
    )
    validate.add_argument("--config", required=True, metavar="PATH", help="YAML configuration file")
    validate.add_argument("--json", action="store_true", help="print the resolved configuration")

    for name in PLANNED_COMMANDS:
        commands.add_parser(
            name,
            help="planned; not implemented",
            description=f"{name} is not implemented in this version.",
            allow_abbrev=False,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return a process exit status; never start an interactive menu."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "doctor":
        return _doctor(args.json)
    if args.command == "config":
        return _validate_config(args.config, args.json)
    print(
        f"rptu: error: '{args.command}' is not implemented in this version. "
        "Use 'rptu doctor' or 'rptu config validate --config PATH' for the available commands.",
        file=sys.stderr,
    )
    return 2
