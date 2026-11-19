#!/usr/bin/env python3
"""Replay a recorded evaluation trace through a live dashboard.

No model weights, no Zoom credentials, no GPU. This reads the per-second
prediction traces produced by the real evaluation runs in results_diagnostics/
and streams them into a browser dashboard at the speed they were captured,
alongside the ground-truth labels and a running AUROC.

It exists so the system can be demonstrated on any machine: the live pipeline
(demo/feed_video.py) needs the fine-tuned DF40 checkpoints, this does not.

Standard library only.
"""

import argparse
import csv
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TRACE_DIR_NAME = Path("experiments") / Path("results_diagnostics") / "traces"
ROC_10S_CSV = Path("experiments") / Path("results_roc_10s_scope_summaries_2026-04-20") / "roc_10s_test_summary.csv"
OFFLINE_CSV = Path("experiments") / Path("results_offline_ft_all_full_2026-04-17") / "offline_results_summary.csv"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def trace_dir() -> Path:
    return repo_root() / TRACE_DIR_NAME


def list_traces():
    directory = trace_dir()
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.csv")):
        stem = path.stem
        if "__" not in stem:
            continue
        backend, run = stem.split("__", 1)
        entries.append({"backend": backend, "run": run, "file": path.name})
    return entries


def read_trace(filename: str):
    path = trace_dir() / filename
    if not path.is_file() or path.suffix != ".csv":
        raise FileNotFoundError(filename)
    points = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                points.append({
                    "second": int(float(row["second"])),
                    "score": float(row["score_oriented"] or row["score_raw"]),
                    "gt": int(float(row["gt_label"])),
                    "alert": int(float(row.get("alert_state") or 0)),
                    "missing": int(float(row.get("missing") or 0)),
                })
            except (TypeError, ValueError, KeyError):
                continue
    return points


def read_summary():
    """Offline (session-pack) vs live (in-pipeline) AUROC per backend, 10s windows, test scope."""
    live = {}
    roc_path = repo_root() / ROC_10S_CSV
    if roc_path.is_file():
        with roc_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    live[row["backend"]] = {
                        "live_auc": float(row["live_auc"]),
                        "offline_auc": float(row["offline_auc"]),
                        "live_windows": int(row["live_windows"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
    offline = {}
    off_path = repo_root() / OFFLINE_CSV
    if off_path.is_file():
        with off_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    offline[row["backend"]] = float(row["auc_test"])
                except (KeyError, TypeError, ValueError):
                    continue
    backends = sorted(set(live) | set(offline),
                      key=lambda b: -(live.get(b, {}).get("offline_auc") or 0))
    rows = []
    for backend in backends:
        entry = live.get(backend, {})
        rows.append({
            "backend": backend,
            "frame_test_auc": offline.get(backend),
            "window_offline_auc": entry.get("offline_auc"),
            "window_live_auc": entry.get("live_auc"),
            "windows": entry.get("live_windows"),
        })
    return rows


def auroc(scores, labels):
    """Mann-Whitney U / rank-based AUROC with tie correction."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    positives = sum(1 for _, l in pairs if l == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(pairs, key=lambda item: item[0])
    ranks = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1
    positive_rank_sum = sum(r for r, (_, l) in zip(ranks, ordered) if l == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deepfake Detection - Trace Replay</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --txt:#e6e8ee;
          --muted:#98a0b0; --fake:#ff6b6b; --real:#4ecdc4; --accent:#7aa2f7; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:16px; }
  h1 { font-size:18px; margin:0 0 2px; }
  p.sub { color:var(--muted); margin:0 0 16px; font-size:13px; }
  .wrap { max-width:1000px; margin:0 auto; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:14px; margin-bottom:14px; }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  select,button { background:#0f1218; color:var(--txt); border:1px solid var(--line);
                  border-radius:7px; padding:7px 10px; font-size:13px; }
  button { cursor:pointer; }
  button.primary { background:var(--accent); color:#0b0d12; border-color:var(--accent); font-weight:600; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }
  .tile { background:#0f1218; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .tile .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .tile .v { font-size:20px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:2px; }
  canvas { width:100%; height:260px; display:block; }
  .legend { display:flex; gap:16px; color:var(--muted); font-size:12px; margin-top:8px; flex-wrap:wrap; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  .note { color:var(--muted); font-size:12px; }
  @media (max-width:520px){ canvas{height:200px;} }
</style></head><body><div class="wrap">
  <h1>Real-time deepfake detection &mdash; trace replay</h1>
  <p class="sub">Per-second detector output from a recorded evaluation run, replayed at capture speed.</p>

  <div class="panel"><div class="row">
    <select id="trace"></select>
    <select id="speed">
      <option value="1">1&times;</option>
      <option value="4" selected>4&times;</option>
      <option value="10">10&times;</option>
      <option value="0">instant</option>
    </select>
    <button class="primary" id="play">Play</button>
    <button id="reset">Reset</button>
  </div></div>

  <div class="panel"><canvas id="chart"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--accent)"></i>detector score</span>
      <span><i class="sw" style="background:var(--fake)"></i>ground truth: fake</span>
      <span><i class="sw" style="background:var(--real)"></i>ground truth: real</span>
      <span>dashed line = alert threshold (0.5)</span>
    </div>
  </div>

  <div class="panel tiles">
    <div class="tile"><div class="k">Elapsed</div><div class="v" id="t">0s</div></div>
    <div class="tile"><div class="k">Score</div><div class="v" id="s">&mdash;</div></div>
    <div class="tile"><div class="k">Ground truth</div><div class="v" id="g">&mdash;</div></div>
    <div class="tile"><div class="k">AUROC (run)</div><div class="v" id="a">&mdash;</div></div>
    <div class="tile"><div class="k">Alerts raised</div><div class="v" id="al">0</div></div>
  </div>
  <div class="panel">
    <div style="font-weight:600;margin-bottom:2px;">Offline vs in-pipeline AUROC</div>
    <p class="note" style="margin:0 0 10px;">Test split, 10-second windows. The gap between the two
      right-hand columns is the cost of moving a lab-validated detector into a live
      video-conferencing pipeline &mdash; the finding this project set out to measure.</p>
    <div style="overflow-x:auto;"><table id="cmp" style="width:100%;border-collapse:collapse;font-size:13px;">
    </table></div>
  </div>
  <p class="note" id="foot"></p>
</div>
<script>
const $ = id => document.getElementById(id);
let points = [], shown = 0, timer = null, runAuroc = null;

function fmt(v, n=3){ return v==null ? "\\u2014" : (+v).toFixed(n); }

function draw(){
  const c = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr;
  const x = c.getContext("2d"); x.scale(dpr,dpr); x.clearRect(0,0,w,h);
  const padL=38, padB=22, padT=10, padR=8;
  const pw = w-padL-padR, ph = h-padT-padB;
  const n = Math.max(points.length,1);
  const X = i => padL + (i/(n-1||1))*pw;
  const Y = v => padT + (1-Math.max(0,Math.min(1,v)))*ph;

  for(let i=0;i<shown;i++){
    const p = points[i]; if(p.gt==null) continue;
    x.fillStyle = p.gt===1 ? "rgba(255,107,107,.16)" : "rgba(78,205,196,.13)";
    x.fillRect(X(i), padT, Math.max(1,pw/(n-1||1)+.6), ph);
  }
  x.strokeStyle="#2a2f3a"; x.lineWidth=1;
  [0,.5,1].forEach(v=>{ x.beginPath(); x.setLineDash(v===.5?[4,4]:[]);
    x.moveTo(padL,Y(v)); x.lineTo(w-padR,Y(v)); x.stroke(); });
  x.setLineDash([]);
  x.fillStyle="#98a0b0"; x.font="11px -apple-system,sans-serif"; x.textAlign="right";
  [0,.5,1].forEach(v=>x.fillText(v.toFixed(1), padL-6, Y(v)+3));

  x.strokeStyle="#7aa2f7"; x.lineWidth=1.8; x.beginPath();
  for(let i=0;i<shown;i++){ const p=points[i]; if(i===0) x.moveTo(X(i),Y(p.score)); else x.lineTo(X(i),Y(p.score)); }
  x.stroke();
  if(shown>0){ const p=points[shown-1];
    x.fillStyle="#7aa2f7"; x.beginPath(); x.arc(X(shown-1),Y(p.score),3.2,0,7); x.fill(); }
}

function stats(){
  if(!shown){ $("t").textContent="0s"; $("s").textContent="\\u2014";
    $("g").textContent="\\u2014"; $("al").textContent="0"; return; }
  const p = points[shown-1];
  $("t").textContent = p.second + "s";
  $("s").textContent = fmt(p.score);
  $("g").textContent = p.gt===1 ? "fake" : "real";
  $("g").style.color = p.gt===1 ? "var(--fake)" : "var(--real)";
  $("al").textContent = points.slice(0,shown).filter(q=>q.alert===1).length;
  $("a").textContent = fmt(runAuroc);
}

function tick(){
  const mult = +$("speed").value;
  if(mult===0){ shown = points.length; draw(); stats(); stop(); return; }
  shown = Math.min(shown+1, points.length);
  draw(); stats();
  if(shown>=points.length) stop();
}
function stop(){ if(timer){ clearInterval(timer); timer=null; } $("play").textContent="Play"; }
function play(){
  if(timer){ stop(); return; }
  if(shown>=points.length) shown=0;
  const mult = +$("speed").value;
  if(mult===0){ tick(); return; }
  $("play").textContent="Pause";
  timer = setInterval(tick, 1000/mult);
}

async function load(){
  const file = $("trace").value;
  stop(); shown = 0;
  const r = await fetch("/api/trace?file="+encodeURIComponent(file));
  const d = await r.json();
  points = d.points; runAuroc = d.auroc;
  $("foot").textContent = d.backend + " \\u00b7 " + d.run + " \\u00b7 "
    + points.length + " seconds \\u00b7 run AUROC " + fmt(d.auroc)
    + " \\u00b7 replayed from experiments/results_diagnostics/traces/" + file;
  draw(); stats();
}

async function loadSummary(){
  const r = await fetch("/api/summary"); const rows = await r.json();
  if(!rows.length) return;
  const cell = (v,bold) => `<td style="padding:7px 10px;text-align:right;font-variant-numeric:tabular-nums;${bold?"font-weight:600;":""}">${v==null?"\u2014":(+v).toFixed(3)}</td>`;
  $("cmp").innerHTML =
    `<thead><tr style="color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;">
      <th style="text-align:left;padding:7px 10px;">Backend</th>
      <th style="text-align:right;padding:7px 10px;">Frame-level (offline)</th>
      <th style="text-align:right;padding:7px 10px;">10s window (offline)</th>
      <th style="text-align:right;padding:7px 10px;">10s window (live)</th>
      <th style="text-align:right;padding:7px 10px;">Drop</th>
    </tr></thead><tbody>` +
    rows.map(r => {
      const drop = (r.window_offline_auc!=null && r.window_live_auc!=null)
        ? (r.window_offline_auc - r.window_live_auc) : null;
      return `<tr style="border-top:1px solid var(--line);">
        <td style="padding:7px 10px;">${r.backend}</td>
        ${cell(r.frame_test_auc)}${cell(r.window_offline_auc)}${cell(r.window_live_auc,true)}
        <td style="padding:7px 10px;text-align:right;color:${drop&&drop>0.15?"var(--fake)":"var(--muted)"};font-variant-numeric:tabular-nums;">${drop==null?"\u2014":(drop<0?"+"+(-drop).toFixed(3):"\u2212"+drop.toFixed(3))}</td>
      </tr>`; }).join("") + "</tbody>";
}

(async function init(){
  await loadSummary();
  const r = await fetch("/api/list"); const list = await r.json();
  $("trace").innerHTML = list.map(t =>
    `<option value="${t.file}">${t.backend} \\u2014 ${t.run}</option>`).join("");
  const preferred = list.findIndex(t => t.backend==="f3net" && /mixed/.test(t.run));
  if(preferred >= 0) $("trace").selectedIndex = preferred;
  $("trace").onchange = load; $("play").onclick = play;
  $("reset").onclick = () => { stop(); shown=0; draw(); stats(); };
  window.addEventListener("resize", draw);
  await load();
})();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
            return
        if route.path == "/api/summary":
            self._send(json.dumps(read_summary()).encode(), "application/json")
            return
        if route.path == "/api/list":
            self._send(json.dumps(list_traces()).encode(), "application/json")
            return
        if route.path == "/api/trace":
            requested = parse_qs(route.query).get("file", [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+\.csv", requested or ""):
                self._send(b'{"error":"bad file"}', "application/json", 400)
                return
            try:
                points = read_trace(requested)
            except FileNotFoundError:
                self._send(b'{"error":"not found"}', "application/json", 404)
                return
            backend, _, run = requested[:-4].partition("__")
            payload = {
                "backend": backend,
                "run": run,
                "points": points,
                "auroc": auroc([p["score"] for p in points], [p["gt"] for p in points]),
            }
            self._send(json.dumps(payload).encode(), "application/json")
            return
        self._send(b"not found", "text/plain", 404)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--list", action="store_true", help="Print available traces and exit.")
    args = parser.parse_args()

    traces = list_traces()
    if args.list:
        for entry in traces:
            print(f"{entry['backend']:<18} {entry['run']:<28} {entry['file']}")
        return 0
    if not traces:
        print(f"No traces found in {trace_dir()}", flush=True)
        return 1

    print(f"Loaded {len(traces)} recorded traces from {trace_dir().relative_to(repo_root())}")
    print(f"Dashboard: http://{args.host}:{args.port}/   (Ctrl+C to stop)")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
