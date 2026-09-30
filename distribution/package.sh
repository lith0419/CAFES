#!/usr/bin/env bash

set -euo pipefail

# Prevent macOS tar from emitting AppleDouble ._* metadata files.
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/dist"
PACKAGE_NAME=""
INCLUDE_RUNS=0

usage() {
  cat <<'EOF'
Usage: ./distribution/package.sh [options]

Create a research/development source snapshot from the current working tree.
This is not the Python sdist produced by `python -m build`. The repository is
not modified; filtering happens in a temporary staging directory.

Options:
  --output-dir DIR  Archive destination (default: ./dist).
  --name NAME       Archive base name without .tar.gz.
  --include-runs    Include runs/ calculation data (excluded by default).
  --help            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "[package] --output-dir needs a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "[package] --name needs a value" >&2; exit 2; }
      PACKAGE_NAME="$2"
      shift
      ;;
    --include-runs)
      INCLUDE_RUNS=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[package] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/pyproject.toml")"
VERSION="${VERSION:-0.0.0}"
STAMP="$(date -u '+%Y%m%d-%H%M%S')"
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
    TREE_STATE="dirty"
  else
    TREE_STATE="clean"
  fi
else
  COMMIT="nogit"
  TREE_STATE="unknown"
fi

if [[ -z "$PACKAGE_NAME" ]]; then
  PACKAGE_NAME="pyscf-agent-${VERSION}-${STAMP}-${COMMIT}-${TREE_STATE}"
fi
if [[ ! "$PACKAGE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "[package] --name may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ARCHIVE_PATH="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

STAGING_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/pyscf-agent-package.XXXXXX")"
STAGING_ROOT="$STAGING_PARENT/$PACKAGE_NAME"
mkdir -p "$STAGING_ROOT"
cleanup() {
  rm -rf -- "$STAGING_PARENT"
}
trap cleanup EXIT

EXCLUDES=(
  './.git'
  './.pyscf-agent'
  './tmp'
  './.venv'
  './venv'
  './build'
  './dist'
  './datasets/QM9_CCSD_upload*'
  './datasets/qm9-ccsd500-v2-dict'
  './docs/si'
  './manuscript'
  './node_modules'
  './agent_knowledge/node_modules'
  './agent_knowledge/wiki'
  './agent_knowledge/dist'
  './agent_knowledge/.llmwiki/candidates'
  './agent_knowledge/.llmwiki/rule-candidates'
  './agent_knowledge/.llmwiki/export'
  './agent_knowledge/.llmwiki/review-export'
  './agent_knowledge/.llmwiki/state.json'
  './agent_knowledge/.llmwiki/last-lint.json'
  './.pytest_cache'
  './.mypy_cache'
  './.ruff_cache'
  './.eggs'
  './__pycache__'
  './htmlcov'
  './.coverage'
  './.DS_Store'
  '._*'
  '*/._*'
  '*/__pycache__'
  '*/.pytest_cache'
  '*.egg-info'
  '*.pyc'
  '*.pyo'
)
if [[ "$INCLUDE_RUNS" -ne 1 ]]; then
  EXCLUDES+=('./runs')
fi
if [[ "$OUTPUT_DIR" == "$ROOT_DIR" ]]; then
  EXCLUDES+=("./$PACKAGE_NAME.tar.gz" "./$PACKAGE_NAME.tar.gz.sha256")
elif [[ "$OUTPUT_DIR" == "$ROOT_DIR"/* ]]; then
  EXCLUDES+=("./${OUTPUT_DIR#"$ROOT_DIR"/}")
fi

COPY_ARGS=(-cf -)
for pattern in "${EXCLUDES[@]}"; do
  COPY_ARGS+=("--exclude=$pattern")
done

echo "[package] staging current working tree"
(
  cd "$ROOT_DIR"
  tar "${COPY_ARGS[@]}" .
) | (
  cd "$STAGING_ROOT"
  tar -xf -
)

# Keep the private root-level environment out of the source snapshot.
rm -f -- "$STAGING_ROOT/llm.env"

cat > "$STAGING_ROOT/DISTRIBUTION_MANIFEST.txt" <<EOF
Package: pyscf-agent
Version: $VERSION
Created UTC: $STAMP
Git commit: $COMMIT
Working tree: $TREE_STATE
Distribution type: source snapshot
Python requirement: >=3.10
Runs included: $([[ "$INCLUDE_RUNS" -eq 1 ]] && printf 'yes' || printf 'no')

This archive was built from the current working tree. Local caches, virtual
environments, Git metadata, build outputs, and private Slurm configuration were
excluded. Private configuration is also excluded; packaged templates under
pyscf_agent/resources/templates are included. Start with:
python configure.py check
EOF

echo "[package] creating $ARCHIVE_PATH"
tar -C "$STAGING_PARENT" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$OUTPUT_DIR"
    sha256sum "$(basename "$ARCHIVE_PATH")" > "$(basename "$CHECKSUM_PATH")"
  )
elif command -v shasum >/dev/null 2>&1; then
  (
    cd "$OUTPUT_DIR"
    shasum -a 256 "$(basename "$ARCHIVE_PATH")" > "$(basename "$CHECKSUM_PATH")"
  )
else
  echo "[package] no SHA-256 tool found; checksum was not written" >&2
  CHECKSUM_PATH=""
fi

echo "[package] verifying exclusions"
ARCHIVE_LIST="$(tar -tzf "$ARCHIVE_PATH")"
for forbidden in '/.git/' '/.pyscf-agent/' '/__pycache__/' '/runs/' '/._'; do
  if [[ "$INCLUDE_RUNS" -eq 1 && "$forbidden" == '/runs/' ]]; then
    continue
  fi
  if printf '%s\n' "$ARCHIVE_LIST" | grep -Fq "$forbidden"; then
    echo "[package] archive contains excluded path: $forbidden" >&2
    exit 1
  fi
done

echo "[package] archive: $ARCHIVE_PATH"
if [[ -n "$CHECKSUM_PATH" ]]; then
  echo "[package] checksum: $CHECKSUM_PATH"
fi
