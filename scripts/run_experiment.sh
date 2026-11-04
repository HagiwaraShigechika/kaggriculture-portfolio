#!/bin/sh
# Run the existing evaluator from the project root, regardless of caller cwd.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/evaluate.py" "$@"
