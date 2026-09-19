#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m jev_eval generate --task all --n "${1:-50}" --out data/seed --seed 1
python3 -m jev_eval validate data/seed
