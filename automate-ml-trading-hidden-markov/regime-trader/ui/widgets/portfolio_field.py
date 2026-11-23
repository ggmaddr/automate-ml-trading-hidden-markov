"""3D "portfolio field": an abstract strand ellipsoid whose geometry is driven by REAL normalized
return paths (the backtest's portfolio and per-symbol equity curves).

Rendering is a dependency-free Canvas2D renderer inside a Streamlit custom component (no CDN, works
offline): perspective projection, drag/keyboard rotation, slow auto-rotate that is disabled under
`prefers-reduced-motion`. `build_strands` is pure and unit-tested; the JS only draws what it is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui import tokens as t

N_STRANDS = 96
N_POINTS = 48

HTML = """
<div class="field" data-testid="portfolio-field">
  <canvas tabindex="0" role="img" aria-label="Abstract 3D geometry of normalized return paths. Drag or use arrow keys to rotate."></canvas>
</div>
"""

CSS = """
:host { display: block; }
.field { width: 100%; height: 520px; position: relative; }
canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; outline: none; border-radius: 6px; }
canvas:active { cursor: grabbing; }
canvas:focus-visible { outline: 2px solid #4A838C; outline-offset: 2px; }
"""

JS = """
export default function (component) {
  const { data, parentElement } = component;
  const canvas = parentElement.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const strands = data.strands || [];
  const teal = data.teal || [74, 131, 140], sand = data.sand || [198, 181, 142];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let angle = 0.6, tilt = -0.32, dragging = false, lastX = 0, lastY = 0, raf = 0, w = 0, h = 0;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  function resize() {
    const r = canvas.getBoundingClientRect();
    w = Math.max(1, r.width); h = Math.max(1, r.height);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function project(x, y, z) {
    // rotate about Y (angle), then tilt about X
    const ca = Math.cos(angle), sa = Math.sin(angle);
    let x1 = x * ca + z * sa, z1 = -x * sa + z * ca;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const y2 = y * ct - z1 * st, z2 = y * st + z1 * ct;
    const f = 3.2, s = f / (f + z2);
    const scale = Math.min(w * 0.5, h * 0.46);
    return [w / 2 + x1 * s * scale, h / 2 - y2 * s * scale, z2];
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    const N = strands.length;
    const items = [];
    for (let k = 0; k < N; k++) {
      const p = strands[k], M = p.length;
      const phi = (2 * Math.PI * k) / N;
      const twist = 1.7 * Math.PI;
      const mix = (k / Math.max(1, N - 1));
      const up = p[M - 1] >= 0;
      const base = up ? teal : sand;
      const other = up ? sand : teal;
      const col = base.map((c, i) => Math.round(c * (1 - 0.35 * mix) + other[i] * 0.35 * mix));
      let prev = null, depth = 0;
      const pts = [];
      for (let j = 0; j < M; j++) {
        const u = j / (M - 1);
        const body = Math.sqrt(Math.max(0, 1 - Math.pow(2 * u - 1, 2)));
        const r = 0.62 * body * (1 + 0.24 * p[j]);
        const th = phi + twist * u;
        const q = project(r * Math.cos(th), (u - 0.5) * 2.0, r * Math.sin(th));
        pts.push(q); depth += q[2];
      }
      items.push({ pts, col, depth: depth / M });
    }
    items.sort((a, b) => b.depth - a.depth);   // far strands first
    for (const it of items) {
      for (let j = 1; j < it.pts.length; j++) {
        const a = it.pts[j - 1], b = it.pts[j];
        const alpha = 0.18 + 0.5 * (1 - (a[2] + 1.2) / 2.4);
        ctx.strokeStyle = `rgba(${it.col[0]},${it.col[1]},${it.col[2]},${Math.max(0.08, Math.min(0.75, alpha))})`;
        ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
    }
  }

  function tick() {
    if (!dragging && !reduced) angle += 0.0035;
    draw();
    raf = requestAnimationFrame(tick);
  }

  const down = (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); };
  const move = (e) => {
    if (!dragging) return;
    angle += (e.clientX - lastX) * 0.008; tilt = Math.max(-1.2, Math.min(1.2, tilt + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY; if (reduced) draw();
  };
  const up = () => { dragging = false; };
  const key = (e) => {
    const step = 0.12;
    if (e.key === 'ArrowLeft') angle -= step; else if (e.key === 'ArrowRight') angle += step;
    else if (e.key === 'ArrowUp') tilt = Math.max(-1.2, tilt - step * 0.6); else if (e.key === 'ArrowDown') tilt = Math.min(1.2, tilt + step * 0.6);
    else return;
    e.preventDefault(); if (reduced) draw();
  };
  canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up); canvas.addEventListener('keydown', key);
  const ro = new ResizeObserver(resize); ro.observe(canvas);
  resize();
  if (!reduced) raf = requestAnimationFrame(tick);

  return () => {
    cancelAnimationFrame(raf); ro.disconnect();
    canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', move);
    canvas.removeEventListener('pointerup', up); canvas.removeEventListener('pointercancel', up); canvas.removeEventListener('keydown', key);
  };
}
"""


def build_strands(series: dict[str, pd.Series], n_strands: int = N_STRANDS, points: int = N_POINTS) -> list[list[float]]:
    """Deterministic strands from real equity paths.

    Each strand is a window (varying start and length) of one series, converted to log-growth from its
    own start, resampled to `points`, and scaled by the global maximum so every value is in [-1, 1].
    """
    usable = {k: v.dropna().astype(float) for k, v in series.items() if len(v.dropna()) >= 10}
    if not usable:
        return []
    names = list(usable)
    raw: list[np.ndarray] = []
    for k in range(n_strands):
        s = usable[names[k % len(names)]]
        values = np.log(s.to_numpy() / s.to_numpy()[0])
        n = len(values)
        start = int(((k * 37) % 100) / 100 * 0.45 * n)
        length = max(10, int((0.45 + 0.55 * (((k * 53) % 100) / 100)) * (n - start)))
        window = values[start:start + length]
        window = window - window[0]
        grid = np.linspace(0, len(window) - 1, points)
        raw.append(np.interp(grid, np.arange(len(window)), window))
    peak = max(float(np.max(np.abs(r))) for r in raw) or 1.0
    return [[round(float(v) / peak, 4) for v in r] for r in raw]


def strand_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Plain-data fallback shown next to the visual (accessibility + transparency)."""
    rows = []
    for name, s in series.items():
        s = s.dropna()
        if len(s) >= 2:
            rows.append({"series": name, "start": s.index[0].date(), "end": s.index[-1].date(),
                         "total return": f"{s.iloc[-1] / s.iloc[0] - 1:.2%}"})
    return pd.DataFrame(rows)


def _rgb(hex_color: str) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _component():
    # Registered at mount time (not import time): the registry belongs to the running Streamlit
    # runtime, so a module-level registration is lost when a new runtime starts (tests, restarts).
    return st.components.v2.component("rt_portfolio_field", html=HTML, css=CSS, js=JS)


def portfolio_field(series: dict[str, pd.Series], key: str = "portfolio-field") -> None:
    strands = build_strands(series)
    if not strands:
        st.caption("Not enough backtest history to draw the portfolio field.")
        return
    _component()(key=key, data={"strands": strands, "teal": _rgb(t.TEAL), "sand": _rgb(t.SERIES_SAND)})
