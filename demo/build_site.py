#!/usr/bin/env python3
"""Build a fully static, self-contained demo site into site/.

Output is a single index.html with the recorded traces and the offline-vs-live
summary inlined, so it works both on a static host (GitHub Pages, Cloudflare
Pages, Netlify) and by double-clicking the file locally. No server, no build
tooling, no dependencies.

    python3 demo/build_site.py
    open site/index.html
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_trace import auroc, list_traces, read_summary, read_trace, repo_root  # noqa: E402

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real-Time Deepfake Detection</title>
<meta name="description" content="Measuring how much a lab-validated deepfake detector loses inside a live video-conferencing pipeline.">
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --txt:#e6e8ee;
          --muted:#98a0b0; --fake:#ff6b6b; --real:#4ecdc4; --accent:#7aa2f7; }
  :root[data-theme="light"] { --bg:#f6f7f9; --panel:#fff; --line:#e3e6ec; --txt:#1a1d23;
          --muted:#5d6675; --accent:#2f5fd0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:28px 16px 56px; }
  header h1 { font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }
  header p { color:var(--muted); margin:0 0 6px; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:12px; padding:16px; margin:16px 0; }
  h2 { font-size:15px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin:0 0 12px; }
  .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
  select,button { background:var(--bg); color:var(--txt); border:1px solid var(--line);
                  border-radius:8px; padding:8px 11px; font-size:13px; font-family:inherit; }
  button { cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; }
  .tile { background:var(--bg); border:1px solid var(--line); border-radius:9px; padding:11px 13px; }
  .tile .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .tile .v { font-size:21px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:3px; }
  canvas { width:100%; height:270px; display:block; }
  .legend { display:flex; gap:16px; color:var(--muted); font-size:12px; margin-top:10px; flex-wrap:wrap; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em;
       text-align:right; padding:8px 10px; font-weight:600; }
  th:first-child { text-align:left; }
  td { padding:8px 10px; text-align:right; font-variant-numeric:tabular-nums;
       border-top:1px solid var(--line); }
  td:first-child { text-align:left; }
  .note { color:var(--muted); font-size:12.5px; }
  footer { color:var(--muted); font-size:12.5px; margin-top:24px; }
  a { color:var(--accent); }
  a.cta { display:inline-block; border:1px solid var(--accent); border-radius:8px; padding:7px 12px;
          text-decoration:none; font-weight:600; font-size:14px; }
  .scroll { overflow-x:auto; }
  @media (max-width:520px){ canvas{height:210px;} header h1{font-size:21px;} }
</style></head><body><div class="wrap">

<header>
  <h1>Real-time deepfake detection in a live video call</h1>
  <p>A detector that scores <strong>0.863</strong> AUROC on its own test data scores
     <strong>0.659</strong> on the same material through a running video-conferencing
     pipeline. This measures that gap across six architectures.</p>
  <p><a class="cta" href="live.html">Run the detectors on your own camera, in your browser &rarr;</a></p>
  <p class="note">BSc final-year project, University of Manchester &middot; Zoom Video SDK bot (C++) &rarr;
     FastAPI inference server &rarr; live dashboard</p>
</header>

<div class="panel">
  <h2>Offline vs in-pipeline AUROC</h2>
  <p class="sub">Test split, 10-second windows. The right-hand gap is the cost of leaving the lab.</p>
  <div class="scroll"><table id="cmp"></table></div>
</div>

<div class="panel">
  <h2>Per-second detector output</h2>
  <p class="sub">Recorded traces from the evaluation runs, replayed at capture speed against ground truth.</p>
  <div class="row" style="margin-bottom:12px;">
    <select id="trace"></select>
    <select id="speed">
      <option value="1">1&times;</option><option value="4" selected>4&times;</option>
      <option value="10">10&times;</option><option value="0">instant</option>
    </select>
    <button class="primary" id="play">Play</button>
    <button id="reset">Reset</button>
  </div>
  <canvas id="chart"></canvas>
  <div class="legend">
    <span><i class="sw" style="background:var(--accent)"></i>detector score</span>
    <span><i class="sw" style="background:var(--fake)"></i>ground truth: fake</span>
    <span><i class="sw" style="background:var(--real)"></i>ground truth: real</span>
    <span>dashed = alert threshold (0.5)</span>
  </div>
</div>

<div class="panel tiles">
  <div class="tile"><div class="k">Elapsed</div><div class="v" id="t">0s</div></div>
  <div class="tile"><div class="k">Score</div><div class="v" id="s">&mdash;</div></div>
  <div class="tile"><div class="k">Ground truth</div><div class="v" id="g">&mdash;</div></div>
  <div class="tile"><div class="k">AUROC (run)</div><div class="v" id="a">&mdash;</div></div>
  <div class="tile"><div class="k">Alerts</div><div class="v" id="al">0</div></div>
</div>

<div class="panel">
  <h2>Why the per-second signal looks flat</h2>
  <p class="note" style="margin:0;">At one-second granularity the live signal is near chance for every
  backend (0.44&ndash;0.54). Ten-second windowing recovers F3Net to 0.659. The detector is not broken:
  single frames pulled out of a re-encoded, frame-dropped video call carry very little of the signal
  the model was trained on. That is the finding, and it is why the table above reports windows.</p>
</div>

<footer id="foot"></footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const $ = id => document.getElementById(id);
let points = [], shown = 0, timer = null, runAuroc = null;
const fmt = (v,n=3) => v==null ? "\\u2014" : (+v).toFixed(n);

function renderTable(){
  const cell = (v,b) => `<td${b?' style="font-weight:600"':''}>${fmt(v)}</td>`;
  $("cmp").innerHTML = `<thead><tr><th>Backend</th><th>Frame-level<br>(offline)</th>
    <th>10s window<br>(offline)</th><th>10s window<br>(live)</th><th>Drop</th></tr></thead><tbody>` +
    DATA.summary.map(r => {
      const d = (r.window_offline_auc!=null && r.window_live_auc!=null)
        ? r.window_offline_auc - r.window_live_auc : null;
      const col = d!=null && d>0.15 ? "var(--fake)" : "var(--muted)";
      const txt = d==null ? "\\u2014" : (d<0 ? "+"+(-d).toFixed(3) : "\\u2212"+d.toFixed(3));
      return `<tr><td>${r.backend}</td>${cell(r.frame_test_auc)}${cell(r.window_offline_auc)}
        ${cell(r.window_live_auc,1)}<td style="color:${col}">${txt}</td></tr>`;
    }).join("") + "</tbody>";
}

function draw(){
  const c = $("chart"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr;
  const x = c.getContext("2d"); x.setTransform(dpr,0,0,dpr,0,0); x.clearRect(0,0,w,h);
  const padL=40, padB=20, padT=10, padR=8, pw=w-padL-padR, ph=h-padT-padB;
  const n = Math.max(points.length,1);
  const X = i => padL + (i/(n-1||1))*pw;
  const Y = v => padT + (1-Math.max(0,Math.min(1,v)))*ph;
  const band = Math.max(1, pw/(n-1||1)+0.6);
  for(let i=0;i<shown;i++){
    const p = points[i];
    x.fillStyle = p[2]===1 ? "rgba(255,107,107,.17)" : "rgba(78,205,196,.14)";
    x.fillRect(X(i), padT, band, ph);
  }
  const grid = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
  x.strokeStyle = grid || "#2a2f3a"; x.lineWidth = 1;
  [0,.5,1].forEach(v => { x.beginPath(); x.setLineDash(v===.5?[4,4]:[]);
    x.moveTo(padL,Y(v)); x.lineTo(w-padR,Y(v)); x.stroke(); });
  x.setLineDash([]);
  x.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#98a0b0";
  x.font = "11px -apple-system,sans-serif"; x.textAlign = "right";
  [0,.5,1].forEach(v => x.fillText(v.toFixed(1), padL-6, Y(v)+3));
  x.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7aa2f7";
  x.lineWidth = 1.8; x.beginPath();
  for(let i=0;i<shown;i++){ i ? x.lineTo(X(i),Y(points[i][1])) : x.moveTo(X(i),Y(points[i][1])); }
  x.stroke();
  if(shown){ x.fillStyle = x.strokeStyle; x.beginPath(); x.arc(X(shown-1),Y(points[shown-1][1]),3.2,0,7); x.fill(); }
}

function stats(){
  if(!shown){ $("t").textContent="0s"; $("s").textContent="\\u2014";
    $("g").textContent="\\u2014"; $("al").textContent="0"; $("a").textContent=fmt(runAuroc); return; }
  const p = points[shown-1];
  $("t").textContent = p[0]+"s";
  $("s").textContent = fmt(p[1]);
  $("g").textContent = p[2]===1 ? "fake" : "real";
  $("g").style.color = p[2]===1 ? "var(--fake)" : "var(--real)";
  let alerts = 0; for(let i=0;i<shown;i++) if(points[i][3]===1) alerts++;
  $("al").textContent = alerts;
  $("a").textContent = fmt(runAuroc);
}

function stop(){ if(timer){ clearInterval(timer); timer=null; } $("play").textContent="Play"; }
function tick(){
  if(+$("speed").value===0){ shown=points.length; draw(); stats(); stop(); return; }
  shown = Math.min(shown+1, points.length); draw(); stats();
  if(shown>=points.length) stop();
}
function play(){
  if(timer){ stop(); return; }
  if(shown>=points.length) shown = 0;
  const m = +$("speed").value;
  if(m===0){ tick(); return; }
  $("play").textContent = "Pause";
  timer = setInterval(tick, 1000/m);
}

function load(){
  stop(); shown = 0;
  const t = DATA.traces[+$("trace").value];
  points = t.points; runAuroc = t.auroc;
  $("foot").textContent = `${t.backend} \\u00b7 ${t.run} \\u00b7 ${points.length} seconds \\u00b7 `
    + `run AUROC ${fmt(t.auroc)} \\u00b7 replayed from experiments/results_diagnostics/traces/${t.file}`;
  draw(); stats();
}

renderTable();
$("trace").innerHTML = DATA.traces.map((t,i) =>
  `<option value="${i}">${t.backend} \\u2014 ${t.run}</option>`).join("");
const pref = DATA.traces.findIndex(t => t.backend==="f3net" && /mixed/.test(t.run));
$("trace").selectedIndex = pref >= 0 ? pref : 0;
$("trace").onchange = load;
$("play").onclick = play;
$("reset").onclick = () => { stop(); shown=0; draw(); stats(); };
window.addEventListener("resize", draw);
load();
</script></body></html>
"""


def main() -> int:
    root = repo_root()
    out_dir = root / "site"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    traces = []
    for entry in list_traces():
        pts = read_trace(entry["file"])
        if not pts:
            continue
        compact = [[p["second"], round(p["score"], 4), p["gt"], p["alert"]] for p in pts]
        traces.append({
            "backend": entry["backend"],
            "run": entry["run"],
            "file": entry["file"],
            "points": compact,
            "auroc": auroc([p["score"] for p in pts], [p["gt"] for p in pts]),
        })

    payload = {"traces": traces, "summary": read_summary()}
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    (out_dir / "index.html").write_text(html)
    (out_dir / ".nojekyll").write_text("")

    live = Path(__file__).resolve().parent / "live.html"
    if live.is_file():
        shutil.copy2(live, out_dir / "live.html")

    size_kb = (out_dir / "index.html").stat().st_size / 1024
    print(f"Wrote {out_dir / 'index.html'}  ({size_kb:,.0f} KB, self-contained)")
    print(f"  {len(traces)} traces, {len(payload['summary'])} backends")
    if (out_dir / "live.html").exists():
        print(f"  + live.html (browser inference, ONNX Runtime Web)")
    print("  open site/index.html      (works from the filesystem)")
    print("  or publish site/ to GitHub Pages / Cloudflare Pages / Netlify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
