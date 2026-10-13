"""Strict, CPU-only configuration for the public toolkit commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml
from yaml.constructor import ConstructorError


class ConfigError(ValueError):
    """A configuration cannot be read or does not match the supported schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous mappings instead of silently keeping the last value."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in keys
                keys.add(key)
            except TypeError as error:
                raise ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    "mapping keys must be scalar values",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while reading a mapping",
                    node.start_mark,
                    "duplicate key {!r}; keep only one value".format(key),
                    key_node.start_mark,
                )
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    num_workers: int = 0


@dataclass(frozen=True)
class AppConfig:
    schema_version: int
    paths: PathsConfig
    runtime: RuntimeConfig

    def to_dict(self) -> Dict[str, Any]:
        """Return the resolved configuration with JSON-serializable values."""
        return {
            "schema_version": self.schema_version,
            "paths": {
                "input_dir": str(self.paths.input_dir),
                "output_dir": str(self.paths.output_dir),
            },
            "runtime": {
                "seed": self.runtime.seed,
                "num_workers": self.runtime.num_workers,
            },
        }


def _fail(source: Path, field: str, message: str) -> None:
    raise ConfigError("{}: {}: {}".format(source, field, message))


def _mapping(value: Any, source: Path, field: str, allowed) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(source, field, "expected a mapping")
    for key in value:
        if not isinstance(key, str):
            _fail(source, field, "field names must be strings; got {!r}".format(key))
        if key not in allowed:
            location = key if field == "config" else "{}.{}".format(field, key)
            _fail(source, location, "unknown field; allowed fields: {}".format(
                ", ".join(sorted(allowed))
            ))
    return value


def _directory(value: Any, source: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _fail(source, field, "expected a nonempty path string")
    try:
        directory = Path(value).expanduser()
        if not directory.is_absolute():
            directory = source.parent / directory
        return directory.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        _fail(source, field, "cannot resolve path: {}".format(error))


def _integer(value: Any, source: Path, field: str, maximum=None) -> int:
    if type(value) is not int:
        _fail(source, field, "expected an integer (booleans are not accepted)")
    if value < 0 or (maximum is not None and value > maximum):
        interval = "0..{}".format(maximum) if maximum is not None else "0 or greater"
        _fail(source, field, "expected {}".format(interval))
    return value


def load_config(path: Union[str, Path]) -> AppConfig:
    """Read and validate a schema-v1 YAML file without creating any directories.

    Dataset paths are resolved relative to the configuration file, irrespective
    of the caller's working directory. Validation does not inspect datasets or
    require a GPU. Environment variable references remain literal path text.
    """
    try:
        source = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ConfigError("{!s}: config: cannot resolve file path: {}".format(
            path, error
        )) from error

    try:
        with source.open("r", encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = "YAML"
        if mark is not None:
            location += " line {}, column {}".format(mark.line + 1, mark.column + 1)
        detail = getattr(error, "problem", None) or str(error)
        raise ConfigError("{}: {}: {}".format(source, location, detail)) from error
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError("{}: config: cannot read file: {}".format(source, error)) from error

    root = _mapping(document, source, "config", {"schema_version", "paths", "runtime"})
    if "schema_version" not in root:
        _fail(source, "schema_version", "required field; expected 1")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        _fail(source, "schema_version", "expected integer 1")
    if "paths" not in root:
        _fail(source, "paths", "required mapping with input_dir and output_dir")
    paths = _mapping(root["paths"], source, "paths", {"input_dir", "output_dir"})
    for name in ("input_dir", "output_dir"):
        if name not in paths:
            _fail(source, "paths.{}".format(name), "required nonempty path string")
    input_dir = _directory(paths["input_dir"], source, "paths.input_dir")
    output_dir = _directory(paths["output_dir"], source, "paths.output_dir")
    if input_dir == output_dir:
        _fail(source, "paths.output_dir", "must differ from resolved paths.input_dir")

    runtime = _mapping(root.get("runtime", {}), source, "runtime", {"seed", "num_workers"})
    seed = _integer(runtime.get("seed", 0), source, "runtime.seed", maximum=2**32 - 1)
    num_workers = _integer(runtime.get("num_workers", 0), source, "runtime.num_workers")
    return AppConfig(1, PathsConfig(input_dir, output_dir), RuntimeConfig(seed, num_workers))
