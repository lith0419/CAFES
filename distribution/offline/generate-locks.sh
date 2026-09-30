#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_FILE="$SCRIPT_DIR/requirements-linux-x86_64.in"
UV_COMMAND="${UV:-}"
PYTHON_VERSIONS=(3.10 3.11 3.12 3.13 3.14)

usage() {
  cat <<'EOF'
Usage: ./distribution/offline/generate-locks.sh [options]

Resolve fully pinned Linux x86_64 dependency locks for CPython 3.10-3.14.
Resolution requires uv and network access. Only binary wheels compatible with
the version-specific minimum glibc target are accepted. Python 3.15 may be
requested explicitly as a readiness check, but its scientific wheel stack is
not yet complete.

Options:
  --uv PATH             uv executable (default: $UV, uv, or python3 -m uv).
  --python-version X.Y  Generate one version instead of the full matrix.
  --help                Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uv)
      [[ $# -ge 2 ]] || { echo "[lock] --uv needs a value" >&2; exit 2; }
      UV_COMMAND="$2"
      shift
      ;;
    --python-version)
      [[ $# -ge 2 ]] || { echo "[lock] --python-version needs a value" >&2; exit 2; }
      PYTHON_VERSIONS=("$2")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[lock] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -n "$UV_COMMAND" ]]; then
  command -v "$UV_COMMAND" >/dev/null 2>&1 || {
    echo "[lock] uv was not found: $UV_COMMAND" >&2
    exit 2
  }
  UV_RUN=("$UV_COMMAND")
elif command -v uv >/dev/null 2>&1; then
  UV_RUN=(uv)
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import uv' >/dev/null 2>&1; then
  UV_RUN=(python3 -m uv)
else
  echo "[lock] uv was not found" >&2
  echo "[lock] install it with: python -m pip install 'uv>=0.8,<1'" >&2
  exit 2
fi
[[ -f "$INPUT_FILE" ]] || {
  echo "[lock] missing shared input: $INPUT_FILE" >&2
  exit 2
}

for version in "${PYTHON_VERSIONS[@]}"; do
  if [[ ! "$version" =~ ^3\.(10|11|12|13|14|15)$ ]]; then
    echo "[lock] unsupported Python version: $version (expected 3.10-3.15)" >&2
    exit 2
  fi

  compact="${version/.}"
  if [[ "$version" == "3.14" || "$version" == "3.15" ]]; then
    python_platform="x86_64-manylinux_2_28"
    platform_label="manylinux_2_28 / glibc 2.28+"
  else
    python_platform="x86_64-manylinux_2_17"
    platform_label="manylinux2014 / glibc 2.17+"
  fi
  output="$SCRIPT_DIR/requirements-linux-x86_64-py${compact}.lock"
  temporary="$(mktemp "${TMPDIR:-/tmp}/pyscf-agent-py${compact}.XXXXXX")"
  cleanup_lock() {
    rm -f -- "$temporary"
  }
  trap cleanup_lock EXIT

  echo "[lock] resolving CPython $version for $platform_label"
  "${UV_RUN[@]}" pip compile "$INPUT_FILE" \
    --python-version "$version" \
    --python-platform "$python_platform" \
    --only-binary=:all: \
    --resolution highest \
    --no-annotate \
    --no-header \
    --output-file "$temporary"

  {
    echo "# Fully resolved for Linux x86_64, $platform_label, CPython $version."
    echo "# Generated from requirements-linux-x86_64.in by generate-locks.sh."
    echo "# Do not loosen pins in an offline reproducibility bundle."
    cat "$temporary"
  } > "$output"
  rm -f -- "$temporary"
  trap - EXIT
  echo "[lock] wrote $output"
done
