#!/usr/bin/env python3
"""Machine paths in one place, so no script carries an absolute path of its own.

Every dataset root, ROS setup file, interpreter and container mount the pipeline
needs lives in ``config/paths.yaml``. That file is gitignored (``config/*`` is,
like every other machine-specific config in this repo) and
``config/examples/paths.example.yaml`` is the committed template -- which is also
the fallback, so a fresh checkout runs before anyone copies anything.

    from paths import dataset, path, ros_env_overrides

    gt = dataset("hilti_2023")["groundtruth"]     # already expanded, str
    prof = path("ros.dds_profile")                # pathlib.Path

Values may reference other values, which is what keeps a machine to one edit:

    ${VAR}                 environment variable; unset and no fallback -> error
                           naming the key that wanted it, not a bare KeyError
    ${VAR:-fallback}       ... with a fallback, itself expandable
    ${datasets.hilti.raw}  another key in this file, by dotted path
    ${SDP_REPO}            always defined: this checkout
    ~                      the invoking user's home

``python3 src/paths.py`` prints every resolved value and marks the ones missing
from disk; ``--json`` emits the same tree for a shell script to read one key out
of. Run it first whenever a script cannot find something.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"

# config/paths.yaml first, the committed template second. Listing the template as a
# fallback rather than requiring a copy means a clone works immediately, and the
# copy only becomes necessary once a path here does not fit the machine.
CANDIDATES = (CONFIG_DIR / "paths.yaml", CONFIG_DIR / "examples" / "paths.example.yaml")

_cache: dict[str, dict] = {}


class PathsError(RuntimeError):
    """A key could not be resolved. Message names the key and what it wanted."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def config_file(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The paths file in use. ``$SDP_PATHS`` overrides the search."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("SDP_PATHS")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in CANDIDATES:
        if candidate.is_file():
            return candidate
    raise PathsError(f"no paths config found; looked for {' then '.join(map(str, CANDIDATES))}")


def load(explicit: str | os.PathLike[str] | None = None) -> dict:
    """The raw (unexpanded) tree, parsed once per file."""
    file = config_file(explicit)
    key = str(file)
    if key not in _cache:
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise PathsError(f"{file}: top level must be a mapping, got {type(data).__name__}")
        _cache[key] = data
    return _cache[key]


def reload() -> None:
    """Drop the parse cache. For tests that write a config and reload it."""
    _cache.clear()


# --------------------------------------------------------------------------
# Reference expansion
# --------------------------------------------------------------------------
def _close_brace(text: str, start: int) -> int:
    """Index of the ``}`` closing the ``${`` whose body starts at `start`."""
    depth = 0
    i = start
    while i < len(text):
        if text.startswith("${", i):
            depth += 1
            i += 2
            continue
        if text[i] == "}":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    raise PathsError(f"unbalanced ${{ in {text!r}")


def _split_default(body: str) -> tuple[str, str | None]:
    """``VAR:-fallback`` -> ('VAR', 'fallback'), ignoring ``:-`` inside nested refs."""
    depth = 0
    i = 0
    while i < len(body):
        if body.startswith("${", i):
            depth += 1
            i += 2
            continue
        if body[i] == "}":
            depth -= 1
        elif depth == 0 and body.startswith(":-", i):
            return body[:i], body[i + 2:]
        i += 1
    return body, None


def _expand(value: str, cfg: dict, key: str, seen: tuple[str, ...]) -> str:
    out: list[str] = []
    i = 0
    while i < len(value):
        if value.startswith("${", i):
            end = _close_brace(value, i + 2)
            name, default = _split_default(value[i + 2:end])
            out.append(_resolve_ref(name, default, cfg, key, seen))
            i = end + 1
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def _resolve_ref(name: str, default: str | None, cfg: dict, key: str,
                 seen: tuple[str, ...]) -> str:
    if name == "SDP_REPO":
        return str(REPO)
    if "." in name:                      # a reference to another key in this file
        if name in seen:
            raise PathsError(f"{key}: reference cycle through {name} ({' -> '.join(seen)})")
        return _get(cfg, name, seen + (name,))
    env = os.environ.get(name)
    if env is not None:
        return env
    if default is not None:
        return _expand(default, cfg, key, seen)
    raise PathsError(f"{key}: environment variable {name} is not set and has no fallback "
                     f"(edit {config_file()} or export {name})")


def _walk(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise PathsError(f"no such key in {config_file()}: {dotted}")
        node = node[part]
    return node


def _get(cfg: dict, dotted: str, seen: tuple[str, ...] = ()) -> str:
    node = _walk(cfg, dotted)
    if isinstance(node, (dict, list)):
        raise PathsError(f"{dotted} is a {type(node).__name__}, not a value; "
                         f"use section()/get_list()")
    return _expandable(node, cfg, dotted, seen)


def _expandable(node, cfg: dict, dotted: str, seen: tuple[str, ...] = ()) -> str:
    """Expand a scalar leaf; non-strings (numbers, bools) pass through as text."""
    text = node if isinstance(node, str) else str(node)
    expanded = _expand(text, cfg, dotted, seen)
    return os.path.expanduser(expanded)


# --------------------------------------------------------------------------
# Public accessors
# --------------------------------------------------------------------------
_MISSING = object()


def get(dotted: str, default=_MISSING, *, config: str | os.PathLike[str] | None = None) -> str:
    """One expanded value. `default` is returned for a key that is not in the file."""
    cfg = load(config)
    try:
        return _get(cfg, dotted)
    except PathsError:
        if default is _MISSING:
            raise
        return default


def path(dotted: str, *, config: str | os.PathLike[str] | None = None) -> Path:
    return Path(get(dotted, config=config))


def get_list(dotted: str, *, config: str | os.PathLike[str] | None = None) -> list[str]:
    """A list-valued key, every element expanded."""
    cfg = load(config)
    node = _walk(cfg, dotted)
    if not isinstance(node, list):
        raise PathsError(f"{dotted} is not a list")
    return [_expandable(v, cfg, f"{dotted}[{i}]") for i, v in enumerate(node)]


def section(dotted: str, *, config: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """A mapping of scalars, every value expanded. Nested mappings are skipped."""
    cfg = load(config)
    node = _walk(cfg, dotted)
    if not isinstance(node, dict):
        raise PathsError(f"{dotted} is not a mapping")
    return {k: _expandable(v, cfg, f"{dotted}.{k}")
            for k, v in node.items() if not isinstance(v, (dict, list))}


def dataset(name: str, *, config: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Every path for one dataset, e.g. ``dataset("hilti_2023")["groundtruth"]``."""
    return section(f"datasets.{name}", config=config)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _flatten(cfg: dict, prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in cfg.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            items += _flatten(value, f"{dotted}.")
        elif isinstance(value, list):
            items += [(f"{dotted}[{i}]", v) for i, v in enumerate(value)]
        else:
            items.append((dotted, value))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("key", nargs="?", help="print just this key (dotted), unexpanded "
                                          "keys and lists included")
    ap.add_argument("--config", help="paths file to read instead of the default search")
    ap.add_argument("--json", action="store_true", help="emit the resolved tree as JSON")
    a = ap.parse_args(argv)

    file = config_file(a.config)
    cfg = load(a.config)

    if a.key:
        node = _walk(cfg, a.key)
        if isinstance(node, list):
            print("\n".join(get_list(a.key, config=a.config)))
        elif isinstance(node, dict):
            print(json.dumps(section(a.key, config=a.config), indent=2))
        else:
            print(get(a.key, config=a.config))
        return 0

    resolved: dict[str, str] = {}
    problems: list[str] = []
    for dotted, _ in _flatten(cfg):
        base, _, index = dotted.partition("[")
        try:
            if index:
                resolved[dotted] = get_list(base, config=a.config)[int(index[:-1])]
            else:
                resolved[dotted] = get(dotted, config=a.config)
        except PathsError as exc:
            problems.append(str(exc))

    if a.json:
        print(json.dumps(resolved, indent=2))
        return 1 if problems else 0

    print(f"config {file}\n")
    width = max((len(k) for k in resolved), default=0)
    for key, value in resolved.items():
        # Only absolute paths are checkable; a mount point inside a container is not
        # expected to exist on the host, so flag rather than fail.
        mark = ""
        if value.startswith("/") and not key.endswith("_mount"):
            mark = "" if Path(value).exists() else "   <- missing"
        print(f"{key:<{width}}  {value}{mark}")
    for problem in problems:
        print(f"\n! {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
