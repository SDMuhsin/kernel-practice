#!/usr/bin/env bash
# Start JupyterLab rooted at the repo.
#   bash scripts/lab.sh            # port 8888
#   PORT=9000 bash scripts/lab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck disable=SC1091
. "$ROOT/env.sh"
exec jupyter lab --no-browser --ip="${IP:-127.0.0.1}" --port="${PORT:-8888}" \
     --ServerApp.root_dir="$ROOT"
