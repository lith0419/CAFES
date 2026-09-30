#!/bin/sh
# The installed plugin is disposable; its configured runtime and runs are not.
set -eu

config_file=${PYSCF_AGENT_CODEX_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/pyscf-agent/codex-mcp.sh}
if [ ! -f "$config_file" ]; then
    echo "PySCF Agent is not configured. Run scripts/configure.py from this plugin with the Python environment containing pyscf-agent[mcp]. Missing: $config_file" >&2
    exit 2
fi

# Configuration is a locally generated command, executed without a login shell.
# It replaces this process with the existing MCP server; stdout stays protocol-only.
exec /bin/sh "$config_file" "$@"
