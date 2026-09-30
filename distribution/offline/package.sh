#!/usr/bin/env bash

set -euo pipefail

export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/dist"
PYTHON_COMMAND="${PYTHON:-python3}"
PACKAGE_NAME=""
TARGET_PYTHON_VERSION="3.10"

usage() {
  cat <<'EOF'
Usage: ./distribution/offline/package.sh [options]

Build an offline bundle for Linux x86_64 and a supported CPython version.
The build machine needs internet access, but it may run macOS or Linux.

Options:
  --output-dir DIR  Bundle destination (default: ./dist).
  --name NAME       Bundle base name without .tar.gz.
  --python PATH     Python whose pip performs the download.
  --python-version  Target CPython version, 3.10-3.14 (default: 3.10).
  --help            Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || { echo "[offline-package] --output-dir needs a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "[offline-package] --name needs a value" >&2; exit 2; }
      PACKAGE_NAME="$2"
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "[offline-package] --python needs a value" >&2; exit 2; }
      PYTHON_COMMAND="$2"
      shift
      ;;
    --python-version)
      [[ $# -ge 2 ]] || { echo "[offline-package] --python-version needs a value" >&2; exit 2; }
      TARGET_PYTHON_VERSION="$2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[offline-package] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! "$TARGET_PYTHON_VERSION" =~ ^3\.(10|11|12|13|14)$ ]]; then
  echo "[offline-package] --python-version must be between 3.10 and 3.14" >&2
  exit 2
fi
TARGET_PYTHON="${TARGET_PYTHON_VERSION/.}"
TARGET_ABI="cp${TARGET_PYTHON}"
if [[ "$TARGET_PYTHON_VERSION" == "3.14" ]]; then
  TARGET_PLATFORM="manylinux_2_28_x86_64"
  TARGET_PLATFORM_LABEL="manylinux_2_28 / glibc 2.28+"
  TARGET_MIN_GLIBC="2.28"
  PIP_PLATFORM_ARGS=(--platform "$TARGET_PLATFORM" --platform manylinux2014_x86_64)
else
  TARGET_PLATFORM="manylinux2014_x86_64"
  TARGET_PLATFORM_LABEL="manylinux2014 / glibc 2.17+"
  TARGET_MIN_GLIBC="2.17"
  PIP_PLATFORM_ARGS=(--platform "$TARGET_PLATFORM")
fi

PYTHON_PATH="$(command -v "$PYTHON_COMMAND" 2>/dev/null || true)"
[[ -n "$PYTHON_PATH" ]] || {
  echo "[offline-package] Python not found: $PYTHON_COMMAND" >&2
  exit 2
}
if [[ "$PYTHON_PATH" != /* ]]; then
  PYTHON_PATH="$(cd "$(dirname "$PYTHON_PATH")" && pwd)/$(basename "$PYTHON_PATH")"
fi
PYTHON_COMMAND="$PYTHON_PATH"
if ! (
  cd "$SCRIPT_DIR"
  "$PYTHON_COMMAND" -c 'import build.__main__'
) >/dev/null 2>&1; then
  echo "[offline-package] Python package 'build' is required on the online build host" >&2
  echo "[offline-package] install it with: $PYTHON_COMMAND -m pip install 'build>=1.2,<2'" >&2
  exit 2
fi
LOCK_FILE="distribution/offline/requirements-linux-x86_64-py${TARGET_PYTHON}.lock"
for path in distribution/package.sh distribution/offline/install.sh distribution/offline/generate-locks.sh distribution/offline/requirements-linux-x86_64.in "$LOCK_FILE" pyscf_agent/resources/templates/llm.env pyscf_agent/resources/templates/remote.ini pyscf_agent/resources/templates/server-slurm.ini pyproject.toml; do
  [[ -f "$ROOT_DIR/$path" ]] || {
    echo "[offline-package] missing required file: $path" >&2
    exit 2
  }
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
  PACKAGE_NAME="pyscf-agent-${VERSION}-linux-x86_64-py${TARGET_PYTHON}-${STAMP}-${COMMIT}-${TREE_STATE}"
fi
if [[ ! "$PACKAGE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "[offline-package] --name may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ARCHIVE_PATH="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
CHECKSUM_PATH="$ARCHIVE_PATH.sha256"

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pyscf-agent-offline.XXXXXX")"
BUNDLE_ROOT="$TEMP_ROOT/$PACKAGE_NAME"
WHEELHOUSE="$BUNDLE_ROOT/wheelhouse"
SOURCE_DIR="$BUNDLE_ROOT/source"
mkdir -p "$WHEELHOUSE" "$SOURCE_DIR"
cleanup() {
  rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT

SOURCE_NAME="pyscf-agent-${VERSION}-source"
echo "[offline-package] creating source archive"
"$ROOT_DIR/distribution/package.sh" \
  --output-dir "$SOURCE_DIR" \
  --name "$SOURCE_NAME"

BUILD_ROOT="$TEMP_ROOT/build"
mkdir -p "$BUILD_ROOT"
tar -C "$BUILD_ROOT" -xzf "$SOURCE_DIR/$SOURCE_NAME.tar.gz"

echo "[offline-package] building pyscf-agent wheel"
(
  cd "$TEMP_ROOT"
  "$PYTHON_COMMAND" -m build \
    --wheel \
    --outdir "$WHEELHOUSE" \
    "$BUILD_ROOT/$SOURCE_NAME"
)

echo "[offline-package] downloading Linux CPython $TARGET_PYTHON_VERSION wheels"
"$PYTHON_COMMAND" -m pip download \
  --dest "$WHEELHOUSE" \
  --no-deps \
  --only-binary=:all: \
  "${PIP_PLATFORM_ARGS[@]}" \
  --python-version "$TARGET_PYTHON" \
  --implementation cp \
  --abi "$TARGET_ABI" \
  --abi abi3 \
  --abi none \
  --requirement "$ROOT_DIR/$LOCK_FILE"

for pattern in pyscf_agent-*.whl pyscf-*.whl h5py-*.whl spglib-*.whl pillow-*.whl ase-*.whl seekpath-*.whl matplotlib-*.whl langgraph-*.whl; do
  if ! compgen -G "$WHEELHOUSE/$pattern" >/dev/null; then
    echo "[offline-package] required wheel was not downloaded: $pattern" >&2
    exit 1
  fi
done

cp "$SCRIPT_DIR/install.sh" "$BUNDLE_ROOT/install_offline.sh"
cp "$ROOT_DIR/$LOCK_FILE" "$BUNDLE_ROOT/offline-requirements.lock"
cat > "$BUNDLE_ROOT/BUNDLE_TARGET.env" <<EOF
python_version=$TARGET_PYTHON_VERSION
machine=x86_64
minimum_glibc=$TARGET_MIN_GLIBC
platform_label=$TARGET_PLATFORM_LABEL
EOF
cp "$ROOT_DIR/pyscf_agent/resources/templates/llm.env" "$BUNDLE_ROOT/llm.env.example"
mkdir -p "$BUNDLE_ROOT/config"
cp "$ROOT_DIR/pyscf_agent/resources/templates/remote.ini" "$BUNDLE_ROOT/config/remote.ini.example"
cp "$ROOT_DIR/pyscf_agent/resources/templates/server-slurm.ini" "$BUNDLE_ROOT/config/server-slurm.ini.example"

WHEEL_COUNT="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
cat > "$BUNDLE_ROOT/OFFLINE_BUNDLE_MANIFEST.txt" <<EOF
Package: pyscf-agent
Version: $VERSION
Created UTC: $STAMP
Git commit: $COMMIT
Working tree: $TREE_STATE
Target platform: Linux x86_64, $TARGET_PLATFORM_LABEL
Target interpreter: CPython $TARGET_PYTHON_VERSION
Wheel count: $WHEEL_COUNT

This bundle installs without network access and without invoking a Conda
solver. Run ./install_offline.sh --help after extracting it.
EOF

{
  printf '\nLocked wheels:\n'
  find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' -exec basename {} \; | LC_ALL=C sort
} >> "$BUNDLE_ROOT/OFFLINE_BUNDLE_MANIFEST.txt"

(
  cd "$WHEELHOUSE"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum ./*.whl > ../WHEELHOUSE_SHA256SUMS.txt
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 ./*.whl > ../WHEELHOUSE_SHA256SUMS.txt
  else
    echo "[offline-package] no SHA-256 tool found for the wheelhouse" >&2
    exit 1
  fi
)

chmod +x "$BUNDLE_ROOT/install_offline.sh"
echo "[offline-package] creating $ARCHIVE_PATH"
tar -C "$TEMP_ROOT" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"

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
  echo "[offline-package] no SHA-256 tool found; checksum was not written" >&2
  CHECKSUM_PATH=""
fi

echo "[offline-package] bundle: $ARCHIVE_PATH"
echo "[offline-package] wheels: $WHEEL_COUNT"
if [[ -n "$CHECKSUM_PATH" ]]; then
  echo "[offline-package] checksum: $CHECKSUM_PATH"
fi
