#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSUME_YES=0
CHECK_ONLY=0
REMOVE_ENVIRONMENTS=0
REMOVE_RUNS=0
REMOVE_DIST=0
REMOVE_WIKI=0
REMOVE_SCRATCH=0
REQUIRE_GIT_CLEAN=0
TARGETS=()

usage() {
  cat <<'EOF'
Usage: ./clean.sh [options]

Remove generated build products and caches, then verify the selected cleanup
scope. Source changes and private configuration are never discarded.

Options:
  --yes       Do not ask for confirmation.
  --check     Report generated paths without removing them.
  --all       Also remove local virtual environments and Node dependencies.
  --dist      Also remove prepared distribution packages in dist/.
  --wiki      Also remove generated wiki pages and curation state.
  --scratch   Also remove block2-scratch/ solver files.
  --runs      Also remove runs/; refuses if Git bundle backups are present.
  --release   Require a clean Git source tree after generated files are removed.
  --help      Show this help message.

Private llm.env and .pyscf-agent/*.ini configuration files are never removed.
Distribution packages, generated wiki pages and solver scratch are kept by default.
Use --runs explicitly because calculation artifacts may be scientifically
valuable. --release reports source changes but never resets or deletes them.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      ASSUME_YES=1
      ;;
    --check)
      CHECK_ONLY=1
      ;;
    --all)
      REMOVE_ENVIRONMENTS=1
      ;;
    --runs)
      REMOVE_RUNS=1
      ;;
    --dist) REMOVE_DIST=1 ;;
    --wiki) REMOVE_WIKI=1 ;;
    --scratch) REMOVE_SCRATCH=1 ;;
    --release)
      REQUIRE_GIT_CLEAN=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[clean] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

add_target() {
  local candidate="$1"
  local existing
  [[ -e "$candidate" || -L "$candidate" ]] || return 0
  if [[ "${#TARGETS[@]}" -gt 0 ]]; then
    for existing in "${TARGETS[@]}"; do
      [[ "$existing" == "$candidate" ]] && return 0
    done
  fi
  TARGETS+=("$candidate")
}

collect_targets() {
  local path
  TARGETS=()

  for path in \
    "$ROOT_DIR/build" \
    "$ROOT_DIR/htmlcov" \
    "$ROOT_DIR/.coverage" \
    "$ROOT_DIR/coverage.xml" \
    "$ROOT_DIR/.pytest_cache" \
    "$ROOT_DIR/.mypy_cache" \
    "$ROOT_DIR/.ruff_cache" \
    "$ROOT_DIR/.hypothesis" \
    "$ROOT_DIR/.tox" \
    "$ROOT_DIR/.nox" \
    "$ROOT_DIR/.eggs"; do
    add_target "$path"
  done

  if [[ "$REMOVE_DIST" -eq 1 ]]; then
    add_target "$ROOT_DIR/dist"
  fi
  if [[ "$REMOVE_SCRATCH" -eq 1 ]]; then
    add_target "$ROOT_DIR/block2-scratch"
  fi
  if [[ "$REMOVE_WIKI" -eq 1 ]]; then
  for path in \
    "$ROOT_DIR/agent_knowledge/dist" \
    "$ROOT_DIR/agent_knowledge/wiki" \
    "$ROOT_DIR/agent_knowledge/log.md" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/candidates" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/rule-candidates" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/export" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/review-export" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/state.json" \
    "$ROOT_DIR/agent_knowledge/.llmwiki/last-lint.json"; do
    add_target "$path"
  done
  fi

  while IFS= read -r -d '' path; do
    add_target "$path"
  done < <(
    find "$ROOT_DIR" \
      \( \
        -path "$ROOT_DIR/.git" -o \
        -path "$ROOT_DIR/.pyscf-agent" -o \
        -path "$ROOT_DIR/.venv" -o \
        -path "$ROOT_DIR/venv" -o \
        -path "$ROOT_DIR/node_modules" -o \
        -path "$ROOT_DIR/agent_knowledge/node_modules" -o \
        -path "$ROOT_DIR/runs" \
      \) -prune -o \
      -type d \
      \( \
        -name "__pycache__" -o \
        -name ".pytest_cache" -o \
        -name ".mypy_cache" -o \
        -name ".ruff_cache" -o \
        -name "*.egg-info" \
      \) -prune -print0
  )

  while IFS= read -r -d '' path; do
    add_target "$path"
  done < <(
    find "$ROOT_DIR" \
      \( \
        -path "$ROOT_DIR/.git" -o \
        -path "$ROOT_DIR/.pyscf-agent" -o \
        -path "$ROOT_DIR/.venv" -o \
        -path "$ROOT_DIR/venv" -o \
        -path "$ROOT_DIR/node_modules" -o \
        -path "$ROOT_DIR/agent_knowledge/node_modules" -o \
        -path "$ROOT_DIR/runs" \
      \) -prune -o \
      -type f \
      \( \
        -name "*.pyc" -o \
        -name "*.pyo" -o \
        -name ".coverage.*" -o \
        -name ".DS_Store" -o \
        -name "._*" \
      \) -print0
  )

  while IFS= read -r -d '' path; do
    add_target "$path"
  done < <(find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type f -name "remote-*-job.json" -print0)

  if [[ "$REMOVE_ENVIRONMENTS" -eq 1 ]]; then
    add_target "$ROOT_DIR/.venv"
    add_target "$ROOT_DIR/venv"
    add_target "$ROOT_DIR/node_modules"
    add_target "$ROOT_DIR/agent_knowledge/node_modules"
  fi

  if [[ "$REMOVE_RUNS" -eq 1 ]]; then
    add_target "$ROOT_DIR/runs"
  fi
}

print_targets() {
  local target
  for target in "${TARGETS[@]}"; do
    echo "  - ${target#"$ROOT_DIR"/}"
  done
}

report_git_status() {
  local status
  if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[clean] Git status: unavailable"
    return 0
  fi

  status="$(git -C "$ROOT_DIR" status --short --untracked-files=normal)"
  if [[ -z "$status" ]]; then
    echo "[clean] Git source tree: clean"
    return 0
  fi

  echo "[clean] Git source tree still contains source changes:" >&2
  printf '%s\n' "$status" >&2
  echo "[clean] source changes were preserved; commit or resolve them before release packaging" >&2
  [[ "$REQUIRE_GIT_CLEAN" -eq 0 ]]
}

# Reject the whole request before deleting anything when a runs backup exists.
if [[ "$REMOVE_RUNS" -eq 1 && -d "$ROOT_DIR/runs" ]]; then
  backup="$(find "$ROOT_DIR/runs" -type f -name '*.bundle' -print -quit)"
  if [[ -n "$backup" ]]; then
    echo "[clean] refusing --runs: Git bundle backup exists: ${backup#"$ROOT_DIR"/}" >&2
    echo "[clean] archive or relocate the backup separately before removing runs/" >&2
    exit 2
  fi
fi

collect_targets
echo "[clean] project root: $ROOT_DIR"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  if [[ "${#TARGETS[@]}" -eq 0 ]]; then
    echo "[clean] generated paths: clean"
    report_git_status
    exit $?
  fi
  echo "[clean] found ${#TARGETS[@]} generated path(s):"
  print_targets
  exit 1
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "[clean] generated paths: already clean"
  report_git_status
  exit $?
fi

echo "[clean] will remove ${#TARGETS[@]} generated path(s):"
print_targets

if [[ "$REMOVE_RUNS" -eq 1 ]]; then
  echo "[clean] warning: --runs deletes calculation logs and artifacts"
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Continue? [y/N] " answer
  answer="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"
  if [[ "$answer" != "y" && "$answer" != "yes" ]]; then
    echo "[clean] cancelled"
    exit 0
  fi
fi

for target in "${TARGETS[@]}"; do
  rm -rf -- "$target"
done

collect_targets
if [[ "${#TARGETS[@]}" -ne 0 ]]; then
  echo "[clean] cleanup verification failed; ${#TARGETS[@]} generated path(s) remain:" >&2
  print_targets >&2
  exit 1
fi

echo "[clean] generated paths: clean"
report_git_status
