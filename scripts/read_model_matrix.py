#!/usr/bin/env python3
"""Read the model_matrix from a YAML config and emit one JSON object per line.

Usage in bash:

    python scripts/read_model_matrix.py configs/eval.yaml | while IFS= read -r entry; do
        script=$(echo "$entry" | python3 -c "import sys,json; print(json.load(sys.stdin)['script'])")
        ...
    done

Each output line is a JSON object with at least { "script": "..." } and
an optional "env" dict of variable overrides.

With --dry-run, prints a human-readable summary instead of JSON.
"""

import argparse
import json
import sys
from pathlib import Path


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Error: PyYAML is required for model matrix parsing.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the YAML config file")
    parser.add_argument("--section", default="evaluation.model_matrix", help="Dotted path to the model matrix list")
    parser.add_argument("--dry-run", action="store_true", help="Print a human-readable summary instead of JSON")
    parser.add_argument("--count", action="store_true", help="Print only the number of entries")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    data = _load_yaml(args.config)
    parts = args.section.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            if args.count:
                print("0")
                return
            return
        current = current[part]

    if not isinstance(current, list):
        print(f"Error: '{args.section}' is not a list", file=sys.stderr)
        sys.exit(1)

    if args.count:
        print(len(current))
        return

    for i, entry in enumerate(current):
        if not isinstance(entry, dict):
            continue
        if args.dry_run:
            script = entry.get("script", "?")
            env = entry.get("env", {})
            env_str = ", ".join(f"{k}={v}" for k, v in env.items()) if env else "(no overrides)"
            print(f"  [{i+1}] {script}  {env_str}")
        else:
            print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()
