# Optional LLM Configuration

Structured preparation, validation, execution, and result parsing work without
an LLM. To enable optional request drafting and result analysis, initialize a
private shell environment in the application working directory:

```bash
pyscf-agent-configure init-llm
```

Edit `llm.env`, then load it before starting an interface:

```bash
source ./llm.env
pyscf-agent-web --no-browser
```

The template is distributed as the package resource
`pyscf_agent/resources/templates/llm.env`. Generated `llm.env`
files use mode `0600`, are ignored by Git, and are excluded from source and
offline bundles. Variables are inherited by the UI process and are never
copied into `TaskSpec`, Slurm requests, or scientific artifacts.

The Study Planner retrieves relevant compiled Wiki pages using the current
goal and scientific context extracted from StudySpec: methods, solvers,
options, observables, and scan parameters. Study labels, file paths, JSON
container names, and disabled optional features do not become search terms.
Single-letter model parameters `U`, `V`, and `t` remain searchable. Long pages
contribute matching paragraphs or list items with their section headings,
within the existing per-page character budget, instead of only their opening
text. Its LLM input contains those selected excerpts, the current
study, Builder site context, and the StudySpec output schema. The full runtime
registry is neither a retrieval query nor an LLM input. Backend planning still
uses the registry to validate scientific capabilities and execution conditions.
The output schema also remains in the prompt when a provider requires plain
text instead of structured output.

The Study Planner expects a complete StudySpec candidate. A malformed or
incomplete response gets one correction request with the original goal and
draft. If correction fails or the provider is unavailable, the current draft
is retained; no locally guessed sweep replaces it. A revised scan must include
its sweep or cases explicitly. Contract failures are logged to the Web service's
stderr. Restart the service after Python source changes; a browser refresh
reloads only the frontend.
