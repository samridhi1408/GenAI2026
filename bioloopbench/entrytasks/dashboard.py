"""A local progress page for a running entry-task pipeline.

  python -m bioloopbench.entrytasks.dashboard --run generated/entry5k --port 8765

Shows counts only (never task or article text): the current stage, per-topic
funnel, tasks written, token use with an estimated cost, and the runner log.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PRICES = {"gemini-3.8-flash": (0.825, 4.125), "gemini-3.1-pro-preview": (2.0, 12.0)}  # USD per million tokens (in, out)
STOP_AFTER = "2026-11-19"


def _json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _cost(usage):
    total = 0.0
    for model, u in usage.items():
        pin, pout = PRICES.get(model.split(":")[-1], (0, 0))
        total += u.get("input_tokens", 0) / 1e6 * pin + u.get("output_tokens", 0) / 1e6 * pout
    return total


def status(run, target):
    run = Path(run)
    funnel, progress, runner = _json(run / "funnel.json", {}), _json(run / "progress.json", {}), _json(run / "runner.json", {})
    past = sum(_cost(r.get("usage", {})) for r in _json(run / "usage.json", []))
    live = _cost(progress.get("usage", {})) if _alive(runner.get("pid")) else 0.0
    topics, accepted = [], 0
    for tid, f in funnel.items():
        c, e, k = f.get("corpus", {}), f.get("extract", {}), f.get("calibrate", {})
        acc = sum(v for key, v in k.items() if key.startswith("tasks_accepted"))
        accepted += acc
        topics.append({"topic": tid,
                       "articles": c.get("test_articles_selected", 0) + c.get("development_articles_selected", 0),
                       "test_articles": c.get("test_articles_selected", 0),
                       "entries": e.get("entries_seen", 0), "articles_failed": e.get("article_failed", 0),
                       "extracted": sum(v for key, v in e.items() if key.startswith("tasks_extracted")),
                       "rejected": sum(v for key, v in e.items() if key.startswith("tasks_rejected")),
                       "accepted": acc, "accepted_test": k.get("tasks_accepted_test", 0),
                       "symptom_solved": sum(v for key, v in k.items() if key.startswith("symptom_solved")),
                       "calibration_failed": k.get("task_failed", 0)})
    log = (run / "runner.log").read_text().splitlines()[-12:] if (run / "runner.log").exists() else []
    stopped = time.strftime("%Y-%m-%d", time.gmtime()) > STOP_AFTER
    deadline = time.mktime(time.strptime(STOP_AFTER, "%Y-%m-%d")) - time.timezone + 86400  # end of that day, UTC
    return {"now": time.strftime("%Y-%m-%d %H:%M:%S"), "running": _alive(runner.get("pid")), "runner": runner,
            "progress": progress, "topics": topics, "accepted": accepted, "target": target,
            "public_files": len(list((run / "public").glob("*.json"))) if (run / "public").exists() else 0,
            "private_files": len(list((run / "private").glob("*.json"))) if (run / "private").exists() else 0,
            "est_cost_usd": round(past + live, 2), "hours_to_stop": round((deadline - time.time()) / 3600, 1),
            "key_stopped": stopped, "stop_after": STOP_AFTER, "log": log}


PAGE = """<!doctype html><meta charset=utf-8><title>Entry-task run</title>
<style>body{font:15px system-ui;margin:24px auto;max-width:1000px;padding:0 16px;color:#222;background:#fafafa}
h1{font-size:20px}table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #ddd;padding:6px 9px;text-align:right}
td:first-child,th:first-child{text-align:left}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:150px}.card b{display:block;font-size:22px}
.bar{height:14px;background:#e4e4e4;border-radius:7px;overflow:hidden;margin:6px 0 14px}.bar div{height:100%;background:#2a7}
pre{background:#fff;border:1px solid #ddd;padding:10px;overflow:auto}.bad{color:#b00}.ok{color:#080}
@media(prefers-color-scheme:dark){body{background:#181818;color:#ddd}table,.card,pre{background:#222;border-color:#444}td,th{border-color:#444}.bar{background:#333}}</style>
<h1>Entry-task generation <small id=now></small></h1><div id=state></div>
<div class=cards id=cards></div><div>Accepted tasks vs target</div><div class=bar><div id=tbar></div></div>
<div id=stage></div><div class=bar><div id=sbar></div></div>
<table id=t></table><p><small>Per-topic rows fill in when each stage of a topic finishes. Counts only; no task text is shown.</small></p>
<h3>Runner log</h3><pre id=log></pre>
<script>
async function tick(){let s;try{s=await (await fetch('status.json')).json()}catch(e){document.getElementById('state').innerHTML='<b class=bad>dashboard server not reachable</b>';return}
const p=s.progress||{},el=id=>document.getElementById(id);
el('now').textContent='· updated '+s.now;
el('state').innerHTML=s.key_stopped?'<b class=bad>STOPPED: past '+s.stop_after+' UTC, the Gemini key must not be used</b>':(s.running?'<b class=ok>Running</b>':'<b class=bad>Not running</b> (finished, stopped, or the laptop restarted)')+' · stops for good in '+s.hours_to_stop+' h (end of '+s.stop_after+' UTC)';
const card=(k,v)=>'<div class=card>'+k+'<b>'+v+'</b></div>';
el('cards').innerHTML=card('Accepted tasks',s.accepted+' / '+s.target)+card('Task records written',s.private_files)+card('Estimated cost','$'+s.est_cost_usd)+card('Hours left',s.hours_to_stop);
el('tbar').style.width=Math.min(100,100*s.accepted/s.target)+'%';
if(p.stage){const rate=p.done/Math.max(1,(p.updated-p.started)/60),left=rate>0?((p.total-p.done)/rate).toFixed(0):'?';
el('stage').textContent='Now: '+p.topic+' / '+p.stage+' — '+p.done+' of '+p.total+' ('+p.failed+' failed), about '+left+' min left in this stage';
el('sbar').style.width=(100*p.done/Math.max(1,p.total))+'%'}
const cols=['topic','articles','test_articles','entries','articles_failed','extracted','rejected','accepted','accepted_test','symptom_solved','calibration_failed'];
el('t').innerHTML='<tr>'+cols.map(c=>'<th>'+c.replace(/_/g,' ')+'</th>').join('')+'</tr>'+s.topics.map(r=>'<tr>'+cols.map(c=>'<td>'+r[c]+'</td>').join('')+'</tr>').join('');
el('log').textContent=s.log.join('\\n')}
tick();setInterval(tick,10000)</script>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--target", type=int, default=5000)
    args = ap.parse_args(argv)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/status.json"):
                body, kind = json.dumps(status(args.run, args.target)).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"dashboard on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
