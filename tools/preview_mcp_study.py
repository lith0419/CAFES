"""Local MCP Apps test host, not a Codex inline-rendering acceptance.

Run with the MCP Python environment and an existing --study-id. No calculation
tools are exposed. --transport-json accepts `codex mcp get pyscf-agent --json`.
"""
from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys

from mcp import Client, StdioServerParameters


HOST_HTML = '''<!doctype html><html><meta charset="utf-8"><title>Study card protocol preview</title>
<style>body{font:14px system-ui;background:#eef3f1;color:#234139;margin:32px auto;max-width:850px;padding:0 16px}iframe{width:100%;height:650px;border:1px solid #cadbd5;border-radius:14px;background:white}a{color:inherit}</style>
<h2>Study card · protocol preview</h2><p>Local test host connected to the real MCP server. This is not Codex's inline renderer.</p>
<iframe title="Study card" sandbox="allow-scripts"></iframe><p id="events"></p>
<script>
const frame=document.querySelector('iframe');
async function rpc(name) {const r=await fetch('/tool',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});const data=await r.json();if(!r.ok)throw Error(data.error);return data;}
function send(data){frame.contentWindow.postMessage({jsonrpc:'2.0',...data},'*');}
window.addEventListener('message',async event=>{
 if(event.source!==frame.contentWindow||event.data?.jsonrpc!=='2.0')return;
 const data=event.data;document.getElementById('events').textContent=data.method||'Response';
 try{
 if(data.method==='ui/initialize')send({id:data.id,result:{protocolVersion:'2026-01-26',hostInfo:{name:'local-test-host',version:'1'},hostCapabilities:{serverTools:{},openLinks:{}},hostContext:{theme:'light'}}});
 else if(data.method==='ui/notifications/initialized')send({method:'ui/notifications/tool-result',params:await rpc('show_study')});
 else if(data.method==='tools/call')send({id:data.id,result:await rpc(data.params.name)});
 else if(data.method==='ui/notifications/size-changed')frame.style.height=data.params.height+'px';
 else if(data.method==='ui/open-link'){const url=new URL(data.params.url);if(url.hostname!=='127.0.0.1'||url.protocol!=='http:')throw Error('Expected local workbench');send({id:data.id,result:{}});location.assign(url.href);}
 }catch(error){send({id:data.id,error:{code:-32000,message:error.message}});}
});
fetch('/card').then(r=>r.text()).then(html=>{frame.srcdoc=html;});
</script></html>'''


async def main(args):
    repo = Path(__file__).resolve().parents[1]
    transport = (json.loads(Path(args.transport_json).read_text())['transport'] if args.transport_json else
                 {'command': sys.executable, 'args': ['-m', 'pyscf_agent.mcp_server', '--work-dir',
                   str(Path(args.work_dir).resolve()), '--executor', 'local'], 'cwd': str(repo)})
    params = StdioServerParameters(command=transport['command'], args=transport.get('args', []),
                                   cwd=transport.get('cwd'), env={**os.environ, **(transport.get('env') or {})})
    async with Client(params) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        uri = tools['show_study'].meta['ui']['resourceUri']
        card = (await client.read_resource(uri)).contents[0].text
        loop = asyncio.get_running_loop()

        class Handler(BaseHTTPRequestHandler):
            def response(self, payload, content_type, status=200):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path not in ('/', '/card'):
                    self.send_error(404)
                    return
                self.response((card if self.path == '/card' else HOST_HTML).encode(), 'text/html; charset=utf-8')

            def do_POST(self):
                try:
                    if self.path != '/tool':
                        raise ValueError('Unknown preview route')
                    data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if data['name'] not in ('show_study', 'refresh_study_view', 'open_workbench'):
                        raise ValueError('Preview only allows viewing, refreshing and opening this Study')
                    result = asyncio.run_coroutine_threadsafe(
                        client.call_tool(data['name'], {'study_id': args.study_id}), loop).result(timeout=120)
                    self.response(result.model_dump_json(by_alias=True).encode(), 'application/json')
                    print(json.dumps({'tool': data['name'], 'study_id': args.study_id, 'error': result.is_error}), flush=True)
                except Exception as exc:
                    self.response(json.dumps({'error': str(exc)}).encode(), 'application/json', 400)

        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        print(f'Preview: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            await asyncio.to_thread(server.serve_forever)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--work-dir', default='runs')
    parser.add_argument('--transport-json')
    parser.add_argument('--port', type=int, default=0)
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
