#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${LLMWIKI_PROVIDER:-}" ]]; then
  if [[ -n "${PYSCF_AGENT_LLM_BASE_URL:-}" && -n "${PYSCF_AGENT_LLM_MODEL:-}" ]]; then
    export LLMWIKI_PROVIDER="openai"
    export LLMWIKI_MODEL="${LLMWIKI_MODEL:-${PYSCF_AGENT_LLM_MODEL}}"
    export OPENAI_BASE_URL="${PYSCF_AGENT_LLM_BASE_URL}"
    export OPENAI_API_KEY="${PYSCF_AGENT_LLM_API_KEY:-sk-local}"

    if [[ -z "${LLMWIKI_REQUEST_TIMEOUT_MS:-}" && "${PYSCF_AGENT_LLM_TIMEOUT:-}" =~ ^[0-9]+$ ]]; then
      export LLMWIKI_REQUEST_TIMEOUT_MS="$((PYSCF_AGENT_LLM_TIMEOUT * 1000))"
    fi
  else
    cat >&2 <<'EOF'
No llmwiki LLM provider is configured.

From the repository root, initialize and load the private PySCF-agent endpoint:
  python -m pyscf_agent.configure init-llm
  source ./llm.env

Then run this script from the same shell.

Or set llmwiki variables directly, for example:
  export LLMWIKI_PROVIDER=ollama
  export LLMWIKI_MODEL=llama3.1
  export OLLAMA_HOST=http://localhost:11434/v1
EOF
    exit 2
  fi
elif [[ -z "${LLMWIKI_MODEL:-}" && -n "${PYSCF_AGENT_LLM_MODEL:-}" ]]; then
  export LLMWIKI_MODEL="${PYSCF_AGENT_LLM_MODEL}"
fi

if [[ "${LLMWIKI_PROVIDER:-}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${PYSCF_AGENT_LLM_API_KEY:-sk-local}"
fi

if [[ $# -gt 0 ]]; then
  exec npx llmwiki compile "$@"
fi

exec npx llmwiki compile --review --lang English
