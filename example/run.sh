#!/usr/bin/env bash
# Run an intrupt example agent with the correct import path.
#
#   ./run.sh finance_agent
#   AEGMIS_OTLP_ENDPOINT=http://localhost:8090 ./run.sh finance_agent
#   ./run.sh                       # list available examples
#
# Why this exists: example/ sits BESIDE the intrupt_py_sdk package, so
# `uvicorn intrupt_py_sdk.example.X:app` fails — Python resolves
# `intrupt_py_sdk` to the OUTER folder, which has no `adapters`/`core`. Running
# the file as a script with the SDK dir on PYTHONPATH imports the package
# correctly, and each example's own `uvicorn.run(app, port=...)` picks the port.
#
# Env (OPENAI_API_KEY, AEGMIS_API_KEY, AEGMIS_BASE_URL, AEGMIS_OTLP_ENDPOINT,
# SLACK_*, etc.) is read from your shell / a local .env by each example.
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "$EXAMPLE_DIR/.." && pwd)"
PY="$SDK_DIR/.venv/bin/python"

usage() {
  echo "usage: $(basename "$0") <example> [python-args...]" >&2
  echo "available examples:" >&2
  for f in "$EXAMPLE_DIR"/*.py; do
    base="$(basename "${f%.py}")"
    [ "$base" = "run" ] && continue
    echo "  - $base" >&2
  done
  exit 1
}

[ "$#" -ge 1 ] || usage
name="${1%.py}"
target="$EXAMPLE_DIR/$name.py"
if [ ! -f "$target" ]; then
  echo "error: no example '$name' (looked for $target)" >&2
  usage
fi
if [ ! -x "$PY" ]; then
  echo "error: SDK venv python not found at $PY" >&2
  echo "       create it first, e.g.: python -m venv $SDK_DIR/.venv && $SDK_DIR/.venv/bin/pip install -e '$SDK_DIR[test]'" >&2
  exit 1
fi
shift

# SDK dir on PYTHONPATH → the inner intrupt_py_sdk package (adapters/core) wins.
export PYTHONPATH="$SDK_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "▶ running $name.py  (PYTHONPATH=$SDK_DIR)" >&2
exec "$PY" "$target" "$@"
