"""Noninteractive commands that do not initialize an ML runtime."""

import argparse
from importlib import metadata
import json
import platform
import sys
from collections.abc import Sequence

from . import __version__


PLANNED_COMMANDS = ("train", "evaluate", "predict", "benchmark")


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
            name: _package_version(name) for name in ("PyYAML", "numpy", "plyfile", "torch", "warpconvnet")
        },
        "capabilities": {
            "config_validation": True,
            "dataset_preparation": True,
            "prepared_dataset_validation": True,
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
        print("CPU dataset preparation: available (PLY requires the io extra)")
        print("Prepared dataset validation: available")
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


def _data_command(args: argparse.Namespace) -> int:
    from .datasets import DatasetError

    try:
        if args.command == "synthetic":
            from .synthetic import generate_synthetic

            path = generate_synthetic(args.output, args.seed)
            report = {"manifest": str(path), "seed": args.seed}
            message = f"Synthetic dataset written: {path}"
        elif args.command == "inspect":
            from .data_commands import inspect_dataset, inspect_prepared_dataset

            if args.prepared is not None:
                report = inspect_prepared_dataset(args.prepared)
                message = (
                    f"Prepared dataset is valid: {len(report['scenes'])} scenes, "
                    f"{report['total_input_points']} input points, "
                    f"{report['total_valid_points']} valid points, "
                    f"{report['total_invalid_points']} invalid rows removed, "
                    f"{report['total_voxels']} voxels "
                    f"(voxel size: {report['voxel_size']} {report['coordinate_unit']})."
                )
            else:
                report = inspect_dataset(args.manifest)
                message = (
                    f"Dataset is valid: {len(report['scenes'])} scenes, "
                    f"{report['total_points']} points, "
                    f"{report['total_invalid_points']} invalid rows to remove."
                )
        else:
            from .data_commands import preprocess_dataset

            report = preprocess_dataset(args.manifest, args.output, args.voxel_size)
            message = f"Prepared {len(report['scenes'])} scenes in {args.output}"
    except DatasetError as error:
        print(f"rptu: error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(message)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rptu",
        description="RPTU AI Toolkit: CPU dataset validation and preparation.",
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

    synthetic = commands.add_parser(
        "synthetic", help="generate a small seeded CPU demonstration dataset", allow_abbrev=False
    )
    synthetic.add_argument("--output", required=True, metavar="DIR", help="new output directory")
    synthetic.add_argument("--seed", type=int, default=0, help="integer seed in 0..4294967295")
    synthetic.add_argument("--json", action="store_true", help="print the generated manifest path")

    inspect = commands.add_parser(
        "inspect", help="validate a raw or prepared dataset and all scene arrays", allow_abbrev=False
    )
    inspect_source = inspect.add_mutually_exclusive_group(required=True)
    inspect_source.add_argument("--manifest", metavar="PATH", help="raw dataset manifest JSON")
    inspect_source.add_argument("--prepared", metavar="PATH", help="prepared dataset metadata JSON")
    inspect.add_argument("--json", action="store_true", help="print scene and label statistics")

    preprocess = commands.add_parser(
        "preprocess", help="filter and voxelize a dataset on the CPU", allow_abbrev=False
    )
    preprocess.add_argument("--manifest", required=True, metavar="PATH", help="dataset manifest JSON")
    preprocess.add_argument("--output", required=True, metavar="DIR", help="new output directory")
    preprocess.add_argument(
        "--voxel-size", type=float, required=True,
        help="positive voxel edge length in the manifest's coordinate unit",
    )
    preprocess.add_argument("--json", action="store_true", help="print the prepared dataset metadata")

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
    if args.command in {"synthetic", "inspect", "preprocess"}:
        return _data_command(args)
    print(
        f"rptu: error: '{args.command}' is not implemented in this version. "
        "Use 'rptu --help' to see the available commands.",
        file=sys.stderr,
    )
    return 2
