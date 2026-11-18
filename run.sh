#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$#" -eq 0 ]; then
    set -- serve
fi

if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHONPATH="$ROOT/src" exec "$ROOT/.venv/bin/python" -m aiforensics "$@"
fi

if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    PYTHON=python3
elif command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
    PYTHON="py -3"
elif command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/" >&2
    exit 1
fi

PYTHONPATH="$ROOT/src" exec $PYTHON -m aiforensics "$@"
