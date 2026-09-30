# MCP Apps Study Card

Call `show_study(study_id)` for a compact view of an existing Study. A host
supporting MCP Apps can display the card inline; ordinary MCP clients still
receive structured data and text. Full planning, review, execution and analysis
remain in the existing workbench and agent.

## Contents And Actions

- Study name, ID, mode, execution status and saved scientific report status.
- Case counts from agent task state, or the saved plan/report when those counts
  are unavailable. Scheduler job counts are separate and can span adaptive stages.
- Up to 20 cases with method, basis, CAS size, saved task status, energy and the
  report's aggregate convergence flag. Missing values remain unreported.
- **Refresh status** calls `refresh_study_view`, an app-only data tool.
- **Open full workbench** calls the existing `open_workbench` tool and asks the
  host to open its URL. A visible link remains if host navigation is unavailable.

Refresh reads saved artifacts and inspects the configured executor. It does not
collect results, create Runs, approve review actions, or advance orchestration.
Saved energies may belong to the result preceding an active retry. The aggregate
convergence flag is not a new assessment of inner DMRG or impurity solver accuracy.
The card has no timer-based polling. All interaction goes through the host's MCP
bridge; it does not access local files or contact the WebUI from the iframe.

## Protocol And Host Boundary

`show_study` declares `_meta.ui.resourceUri = ui://pyscf-agent/study-v1.html`.
The resource has MIME type `text/html;profile=mcp-app`, inline HTML/CSS/JavaScript,
and empty external connection/resource domains. It uses the standard
`ui/initialize` handshake, tool result notifications, `tools/call`, `ui/open-link`,
theme updates and size notifications. Only the render tool carries a UI resource;
the app-only refresh tool returns the same projection without creating a new card.
Breaking resource changes must use a new versioned URI.

The implementation follows the [MCP Apps specification](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
and [OpenAI's UI guide](https://developers.openai.com/plugins/build/chatgpt-ui).
Protocol support does not establish support in a particular desktop host.

As of the September 16 acceptance, MCP discovery/resource reads, the card bridge,
and a local browser test host connected to the installed plugin passed. **Actual
Codex inline rendering is not yet verified.** The ongoing development task has
not loaded the plugin's new tools. After reinstalling, start a new Codex task
and ask: “用 CAFES 展示 Study 20260916-155300-67cdc98e 的卡片，不要执行计算。”
If that host does not render the card, use its structured summary and
`open_workbench` URL. Do not describe a browser preview as an inline-rendering test.

## Reproduce The Browser Preview

Run from the repository with the configured MCP environment and an existing
Study. This starts a temporary localhost test host and submits no calculations:

```sh
.venv/bin/python tools/preview_mcp_study.py \
  --study-id STUDY_ID --work-dir ./runs
```

Open the printed URL. The preview uses a sandboxed iframe, reads the actual MCP
UI resource and forwards only card display, refresh and workbench opening for
the specified Study. To test an installed plugin instead, pass `--transport-json`
with a JSON file produced by `codex mcp get pyscf-agent --json`. Stop the preview
with Ctrl-C. It does not stop an independently running workbench.
