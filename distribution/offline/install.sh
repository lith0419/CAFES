#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELHOUSE="$ROOT_DIR/wheelhouse"
REQUIREMENTS="$ROOT_DIR/offline-requirements.lock"
CHECKSUMS="$ROOT_DIR/WHEELHOUSE_SHA256SUMS.txt"
TARGET_METADATA="$ROOT_DIR/BUNDLE_TARGET.env"
DEFAULT_PREFIX="${HOME}/.venvs/pyscf-agent"
PREFIX="$DEFAULT_PREFIX"
PYTHON_COMMAND="${PYTHON:-python3}"
USE_CURRENT=0
CHECK_SLURM=0
ROLE="local"

usage() {
  cat <<'EOF'
Usage: ./install_offline.sh [options]

Install the bundled Linux x86_64 wheels without network access or dependency
solving. The required Python and glibc versions are recorded in the bundle.

Options:
  --prefix DIR      Create or update a virtual environment at DIR.
                    Default: ~/.venvs/pyscf-agent
  --python PATH     Matching Python used to create the virtual environment.
  --current         Install into the currently active Python/Conda environment.
  --check-slurm     Verify Slurm client commands after installation.
  --role ROLE       Validate local (SSH client) or server (Slurm) commands.
                    Default: local
  --help            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || { echo "[offline-install] --prefix needs a value" >&2; exit 2; }
      PREFIX="$2"
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "[offline-install] --python needs a value" >&2; exit 2; }
      PYTHON_COMMAND="$2"
      shift
      ;;
    --current)
      USE_CURRENT=1
      ;;
    --check-slurm)
      CHECK_SLURM=1
      ;;
    --role)
      [[ $# -ge 2 ]] || { echo "[offline-install] --role needs a value" >&2; exit 2; }
      ROLE="$2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[offline-install] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$ROLE" != "local" && "$ROLE" != "server" ]]; then
  echo "[offline-install] --role must be local or server" >&2
  exit 2
fi
if [[ "$ROLE" == "server" ]]; then
  CHECK_SLURM=1
fi

[[ -d "$WHEELHOUSE" ]] || {
  echo "[offline-install] missing wheelhouse: $WHEELHOUSE" >&2
  exit 2
}
[[ -f "$REQUIREMENTS" ]] || {
  echo "[offline-install] missing requirements: $REQUIREMENTS" >&2
  exit 2
}
[[ -f "$CHECKSUMS" ]] || {
  echo "[offline-install] missing wheelhouse checksums: $CHECKSUMS" >&2
  exit 2
}
[[ -f "$TARGET_METADATA" ]] || {
  echo "[offline-install] missing target metadata: $TARGET_METADATA" >&2
  exit 2
}

TARGET_PYTHON_VERSION="$(awk -F= '$1 == "python_version" { print $2; exit }' "$TARGET_METADATA")"
TARGET_MACHINE="$(awk -F= '$1 == "machine" { print $2; exit }' "$TARGET_METADATA")"
TARGET_MIN_GLIBC="$(awk -F= '$1 == "minimum_glibc" { print $2; exit }' "$TARGET_METADATA")"
if [[ -z "$TARGET_PYTHON_VERSION" || -z "$TARGET_MACHINE" || -z "$TARGET_MIN_GLIBC" ]]; then
  echo "[offline-install] target metadata is incomplete: $TARGET_METADATA" >&2
  exit 2
fi

echo "[offline-install] verifying wheelhouse SHA-256 checksums"
if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$WHEELHOUSE"
    sha256sum --check "$CHECKSUMS"
  )
elif command -v shasum >/dev/null 2>&1; then
  (
    cd "$WHEELHOUSE"
    shasum -a 256 --check "$CHECKSUMS"
  )
else
  echo "[offline-install] no SHA-256 verification command is available" >&2
  exit 2
fi

command -v "$PYTHON_COMMAND" >/dev/null 2>&1 || {
  echo "[offline-install] Python not found: $PYTHON_COMMAND" >&2
  exit 2
}

PLATFORM="$($PYTHON_COMMAND -c 'import platform, sys; libc = platform.libc_ver(); print("{}|{}|{}.{}|{}|{}".format(sys.platform, platform.machine().lower(), sys.version_info[0], sys.version_info[1], libc[0], libc[1]))')"
IFS='|' read -r SYSTEM_NAME MACHINE PYTHON_VERSION LIBC_NAME LIBC_VERSION <<< "$PLATFORM"
if [[ "$SYSTEM_NAME" != "linux" || ! "$MACHINE" =~ ^(x86_64|amd64)$ || "$PYTHON_VERSION" != "$TARGET_PYTHON_VERSION" ]]; then
  echo "[offline-install] this bundle requires Linux $TARGET_MACHINE and CPython $TARGET_PYTHON_VERSION" >&2
  echo "[offline-install] detected: $SYSTEM_NAME $MACHINE Python $PYTHON_VERSION" >&2
  exit 2
fi
if [[ "$LIBC_NAME" != "glibc" ]] || ! "$PYTHON_COMMAND" -c 'import sys; parse = lambda value: tuple(int(part) for part in value.split(".")[:2]); raise SystemExit(0 if parse(sys.argv[1]) >= parse(sys.argv[2]) else 1)' "$LIBC_VERSION" "$TARGET_MIN_GLIBC"; then
  echo "[offline-install] this bundle requires glibc $TARGET_MIN_GLIBC or newer" >&2
  echo "[offline-install] detected: ${LIBC_NAME:-unknown} ${LIBC_VERSION:-unknown}" >&2
  exit 2
fi

if [[ "$USE_CURRENT" -eq 1 ]]; then
  TARGET_PYTHON="$($PYTHON_COMMAND -c 'import sys; print(sys.executable)')"
  echo "[offline-install] installing into current environment: $TARGET_PYTHON"
else
  if [[ -e "$PREFIX" && ! -f "$PREFIX/pyvenv.cfg" ]]; then
    echo "[offline-install] prefix exists but is not a Python virtual environment: $PREFIX" >&2
    exit 2
  fi
  echo "[offline-install] preparing virtual environment: $PREFIX"
  "$PYTHON_COMMAND" -m venv "$PREFIX"
  TARGET_PYTHON="$PREFIX/bin/python"
fi

echo "[offline-install] installing bundled dependencies"
"$TARGET_PYTHON" -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --no-deps \
  --only-binary=:all: \
  --requirement "$REQUIREMENTS"

PROJECT_WHEELS=("$WHEELHOUSE"/pyscf_agent-*.whl)
if [[ "${#PROJECT_WHEELS[@]}" -ne 1 || ! -f "${PROJECT_WHEELS[0]}" ]]; then
  echo "[offline-install] expected exactly one pyscf-agent wheel in $WHEELHOUSE" >&2
  exit 2
fi

echo "[offline-install] installing pyscf-agent"
"$TARGET_PYTHON" -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --only-binary=:all: \
  --no-deps \
  --force-reinstall \
  "${PROJECT_WHEELS[0]}"

"$TARGET_PYTHON" -m pip check
"$TARGET_PYTHON" -c 'import ase, h5py, langgraph, matplotlib, PIL, pyscf, pyscf_agent, seekpath, spglib; print("[offline-install] runtime imports: OK")'

SCRIPTS_DIR="$($TARGET_PYTHON -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
for entry_point in pyscf-agent pyscf-agent-configure pyscf-agent-web pyscf-agent-rpc pyscf-computational-study; do
  if [[ ! -x "$SCRIPTS_DIR/$entry_point" ]]; then
    echo "[offline-install] missing console command: $SCRIPTS_DIR/$entry_point" >&2
    exit 1
  fi
done
echo "[offline-install] console commands: OK at $SCRIPTS_DIR"

if [[ "$CHECK_SLURM" -eq 1 ]]; then
  for command in sbatch srun squeue sacct scancel; do
    command -v "$command" >/dev/null 2>&1 || {
      echo "[offline-install] missing Slurm command: $command" >&2
      exit 1
    }
  done
  echo "[offline-install] Slurm commands: OK"
else
  command -v ssh >/dev/null 2>&1 || {
    echo "[offline-install] missing SSH client command: ssh" >&2
    exit 1
  }
  echo "[offline-install] SSH client command: OK"
fi

echo "[offline-install] installation complete"
if [[ "$USE_CURRENT" -ne 1 ]]; then
  echo "[offline-install] activate with: source $PREFIX/bin/activate"
fi
echo "[offline-install] verify with: $SCRIPTS_DIR/pyscf-agent-configure check --role $ROLE"
