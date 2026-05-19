#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_html(csv_path: Path, duration: float = 200.0, title: str = "Robot trajectory") -> str:
    df = pd.read_csv(csv_path)

    x_col = pick_column(df, ["x", "pos_x", "position_x", "robot_x"])
    y_col = pick_column(df, ["y", "pos_y", "position_y", "robot_y"])
    yaw_col = pick_column(df, ["yaw", "heading", "theta", "orientation", "angle"])
    t_col = pick_column(df, ["t", "time", "time_s", "timestamp", "step"])

    if x_col is None or y_col is None:
        raise KeyError("Нужны колонки координат x/y (или их аналоги).")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float) if t_col else np.arange(len(df), dtype=float)
    yaw = pd.to_numeric(df[yaw_col], errors="coerce").to_numpy(dtype=float) if yaw_col else None

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if yaw is not None:
        mask &= np.isfinite(yaw)
    x, y, t = x[mask], y[mask], t[mask]
    if yaw is not None:
        yaw = yaw[mask]

    order = np.argsort(t, kind="stable")
    x, y, t = x[order], y[order], t[order]
    if yaw is not None:
        yaw = yaw[order]

    _, uniq_idx = np.unique(t, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    x, y, t = x[uniq_idx], y[uniq_idx], t[uniq_idx]
    if yaw is not None:
        yaw = yaw[uniq_idx]

    t = t - t.min()
    tnorm = np.linspace(0, 1, len(t)) if np.allclose(t.max(), 0) else t / t.max()

    waypoints = None
    if "waypoint_idx" in df.columns and "target_x" in df.columns and "target_y" in df.columns:
        w = df[["waypoint_idx", "target_x", "target_y"]].dropna().copy()
        if len(w):
            w = w.groupby("waypoint_idx", as_index=False).first().sort_values("waypoint_idx")
            waypoints = w.rename(columns={"target_x": "x", "target_y": "y"})[["x", "y"]]
    elif "target_x" in df.columns and "target_y" in df.columns:
        w = df[["target_x", "target_y"]].dropna().copy()
        if len(w):
            w = w[(w["target_x"].shift(1) != w["target_x"]) | (w["target_y"].shift(1) != w["target_y"])]
            waypoints = w.rename(columns={"target_x": "x", "target_y": "y"})[["x", "y"]]

    goal_x = float(df["final_x"].dropna().iloc[0]) if "final_x" in df.columns and df["final_x"].notna().any() else None
    goal_y = float(df["final_y"].dropna().iloc[0]) if "final_y" in df.columns and df["final_y"].notna().any() else None

    xs_all = [x]
    ys_all = [y]
    if waypoints is not None and len(waypoints):
        xs_all.append(waypoints["x"].to_numpy())
        ys_all.append(waypoints["y"].to_numpy())
    if goal_x is not None and goal_y is not None:
        xs_all.append(np.array([goal_x]))
        ys_all.append(np.array([goal_y]))

    xmin = min(np.nanmin(a) for a in xs_all)
    xmax = max(np.nanmax(a) for a in xs_all)
    ymin = min(np.nanmin(a) for a in ys_all)
    ymax = max(np.nanmax(a) for a in ys_all)
    pad = max(max(xmax - xmin, ymax - ymin) * 0.08, 1e-6)
    xlim = [xmin - pad, xmax + pad]
    ylim = [ymin - pad, ymax + pad]

    js_data = {
        "x": x.tolist(),
        "y": y.tolist(),
        "t": tnorm.tolist(),
        "yaw": None if yaw is None else yaw.tolist(),
        "waypoints": None if waypoints is None else waypoints.to_dict(orient="records"),
        "goal": None if goal_x is None or goal_y is None else {"x": goal_x, "y": goal_y},
        "xlim": xlim,
        "ylim": ylim,
        "duration": float(duration),
        "title": title,
        "showArrow": True,
        "n": int(len(x)),
    }

    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
html, body {{ margin:0; padding:0; width:100%; height:100%; background:#fff; font-family:Arial, Helvetica, sans-serif; }}
#wrap {{ width:100%; height:100%; position:relative; background:#fff; }}
#plot {{ width:100%; height:100vh; }}
#hud {{
  position:absolute; left:16px; top:16px; z-index:20;
  background:rgba(255,255,255,0.92); border:1px solid #ddd; border-radius:10px;
  padding:10px 12px; font-size:14px; line-height:1.35; color:#111;
  box-shadow:0 2px 10px rgba(0,0,0,0.08); pointer-events:none; white-space:pre-line;
}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<div id="wrap">
  <div id="hud">loading…</div>
  <div id="plot"></div>
</div>

<script>
const DATA = {json.dumps(js_data)};
const gd = document.getElementById('plot');
const hud = document.getElementById('hud');

const x = DATA.x, y = DATA.y, t = DATA.t, yaw = DATA.yaw;
const waypoints = DATA.waypoints || [];
const goal = DATA.goal;
const n = DATA.n;
const cycleMs = DATA.duration * 1000.0;
const frameMs = cycleMs / Math.max(n, 1);

const traces = [];
if (waypoints.length > 0) {{
  traces.push({{
    x: waypoints.map(d => d.x),
    y: waypoints.map(d => d.y),
    mode: 'lines+markers',
    type: 'scatter',
    name: 'waypoints',
    line: {{color: 'rgba(120,120,120,0.45)', width: 2, dash: 'dot'}},
    marker: {{color: 'rgba(120,120,120,0.8)', size: 6}}
  }});
}}
if (goal) {{
  traces.push({{
    x: [goal.x], y: [goal.y],
    mode: 'markers', type: 'scatter',
    name: 'goal',
    marker: {{symbol: 'star', size: 14, color: '#d62728'}}
  }});
}}
traces.push({{
  x: [], y: [],
  mode: 'lines', type: 'scatter',
  name: 'trajectory',
  line: {{color: '#1f77b4', width: 3}}
}});
traces.push({{
  x: [], y: [],
  mode: 'markers', type: 'scatter',
  name: 'robot',
  marker: {{size: 11, color: '#111', line: {{color: '#ffffff', width: 1}}}}
}});
traces.push({{
  x: [], y: [],
  mode: 'lines', type: 'scatter',
  name: 'heading',
  line: {{color: '#111', width: 2}}
}});

const layout = {{
  title: DATA.title,
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  showlegend: true,
  legend: {{orientation:'h', yanchor:'bottom', y:1.02, xanchor:'left', x:0}},
  margin: {{l:70, r:30, t:60, b:60}},
  xaxis: {{ title:'X', range: DATA.xlim, zeroline:false, gridcolor:'rgba(0,0,0,0.08)' }},
  yaxis: {{ title:'Y', range: DATA.ylim, scaleanchor:'x', scaleratio:1, zeroline:false, gridcolor:'rgba(0,0,0,0.08)' }},
}};

const config = {{ responsive:true, displayModeBar:true, scrollZoom:true }};

function updateFrame(i) {{
  const trajX = x.slice(0, i + 1);
  const trajY = y.slice(0, i + 1);
  const robotX = [x[i]];
  const robotY = [y[i]];
  let hx = [x[i], x[i]];
  let hy = [y[i], y[i]];
  if (DATA.showArrow && Array.isArray(yaw) && Number.isFinite(yaw[i])) {{
    const lenX = DATA.xlim[1] - DATA.xlim[0];
    const lenY = DATA.ylim[1] - DATA.ylim[0];
    const arrowLen = Math.max(lenX, lenY) * 0.05;
    hx = [x[i], x[i] + arrowLen * Math.cos(yaw[i])];
    hy = [y[i], y[i] + arrowLen * Math.sin(yaw[i])];
  }}
  const trajIndex = waypoints.length > 0 ? 2 : 1;
  const robotIndex = trajIndex + 1;
  const headingIndex = trajIndex + 2;
  Plotly.restyle(gd, {{x:[trajX], y:[trajY]}}, [trajIndex]);
  Plotly.restyle(gd, {{x:[robotX], y:[robotY]}}, [robotIndex]);
  Plotly.restyle(gd, {{x:[hx], y:[hy]}}, [headingIndex]);
  const elapsed = t[i] * DATA.duration;
  hud.textContent = `t = ${{elapsed.toFixed(1)}} s / ${{DATA.duration.toFixed(1)}} s\nstep = ${{i + 1}} / ${{n}}\nx = ${{x[i].toFixed(3)}}   y = ${{y[i].toFixed(3)}}`;
}}

let lastIndex = -1;
let start = null;
function loop(ts) {{
  if (start === null) start = ts;
  const phase = (ts - start) % cycleMs;
  const i = Math.min(n - 1, Math.floor(phase / frameMs));
  if (i !== lastIndex) {{
    updateFrame(i);
    lastIndex = i;
  }}
  requestAnimationFrame(loop);
}}

Plotly.newPlot(gd, traces, layout, config).then(() => {{
  updateFrame(0);
  requestAnimationFrame(loop);
}});
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(description="Create looping HTML animation for robot trajectory.")
    p.add_argument("--input", required=True, help="CSV with per-step episode data.")
    p.add_argument("--output", required=True, help="Output HTML file.")
    p.add_argument("--duration", type=float, default=200.0, help="Loop duration in seconds.")
    p.add_argument("--title", default="Robot trajectory (looped 200s)", help="Plot title.")
    args = p.parse_args()

    html = build_html(Path(args.input), duration=args.duration, title=args.title)
    Path(args.output).write_text(html, encoding="utf-8")
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
