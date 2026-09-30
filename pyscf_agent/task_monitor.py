"""The same task monitor document is served by HTTP and MCP Apps."""

from importlib.resources import files


def task_monitor_html():
    assets = files('pyscf_agent.web_assets')
    return (assets.joinpath('task-monitor.html').read_text(encoding='utf-8')
            .replace('/* TASK_MONITOR_CSS */', assets.joinpath('task-monitor.css').read_text(encoding='utf-8'))
            .replace('/* TASK_MONITOR_JS */', assets.joinpath('task-monitor.js').read_text(encoding='utf-8')))
