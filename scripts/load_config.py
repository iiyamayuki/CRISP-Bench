#!/usr/bin/env python3
"""Load a YAML config file and emit shell-compatible variable assignments.

Usage in bash scripts:

    eval "$(python scripts/load_config.py configs/nuscenes.yaml)"
    eval "$(python scripts/load_config.py configs/eval.yaml --section evaluation)"

Each leaf value is emitted as:
    KEY=VALUE        (if --export is not set)
    export KEY=VALUE (if --export is set)

By default, config values override any same-named environment variables that
were already present in the current shell. Pass --preserve-env to keep the
existing environment values instead.

Priority order in the stage scripts (high -> low):
    1. Values from the YAML config file
    2. Environment variables loaded before config evaluation
    3. Script-level defaults (handled by the calling bash script)
"""

import argparse
import os
import sys
from pathlib import Path


def _load_yaml(path: str) -> dict:
    """Load YAML using PyYAML if available, otherwise fall back to a minimal parser."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    # Minimal fallback: only supports flat key: value lines and very simple nesting.
    # This is intentionally limited; install PyYAML for full config support.
    data: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                val = val.strip()
                if val:
                    data[key.strip()] = val
    return data


def _navigate(data: dict, section: str) -> dict:
    """Navigate into a dotted section path, e.g. 'evaluation.collection'."""
    parts = section.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            print(f"# Warning: section '{section}' not found in config", file=sys.stderr)
            return {}
        current = current[part]
    if not isinstance(current, dict):
        return {"_value": current}
    return current


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested dict into (KEY, value) pairs with uppercased keys."""
    result = []
    for key, value in d.items():
        full_key = f"{prefix}{key}".upper() if prefix else key.upper()
        if isinstance(value, dict):
            result.extend(_flatten(value, f"{full_key}_"))
        elif isinstance(value, list):
            # Skip list values — they need special handling (e.g. model_matrix)
            continue
        else:
            result.append((full_key, str(value)))
    return result


def _shell_quote(s: str) -> str:
    """Quote a string for safe use in shell assignments."""
    if all(c.isalnum() or c in "._-/~:" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the YAML config file")
    parser.add_argument("--section", default="", help="Dotted path to a subsection (e.g. 'evaluation.collection')")
    parser.add_argument("--export", action="store_true", help="Emit 'export KEY=VALUE' instead of 'KEY=VALUE'")
    parser.add_argument(
        "--preserve-env",
        action="store_true",
        help="Skip config values for variables that already exist in the environment",
    )
    parser.add_argument("--override", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not Path(args.config).is_file():
        print(f"# Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    data = _load_yaml(args.config)
    if args.section:
        data = _navigate(data, args.section)

    pairs = _flatten(data)
    prefix = "export " if args.export else ""

    for key, value in pairs:
        if args.preserve_env and key in os.environ:
            continue
        print(f"{prefix}{key}={_shell_quote(value)}")


if __name__ == "__main__":
    main()
