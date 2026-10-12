"""Plotly figures in the "lab" style: transparent canvas, hairline grid, thin lines,
mono axis labels, palette from ui.tokens. Data prep stays in the views; these only draw."""

from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui import tokens as t
from ui.html import esc

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def style(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=t.MONO, size=11, color=t.LABEL),
        hoverlabel=dict(bgcolor=t.CARD, bordercolor=t.HAIRLINE, font=dict(family=t.MONO, color=t.INK, size=12)),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(family=t.MONO, size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=t.HAIRLINE, tickfont=dict(family=t.MONO, size=10, color=t.LABEL))
    fig.update_yaxes(gridcolor=t.GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(family=t.MONO, size=10, color=t.LABEL))
    return fig


def equity_drawdown(equity: pd.Series, height: int = 480) -> go.Figure:
    from backtest.performance import drawdown_series

    dd = drawdown_series(equity)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color=t.TEAL, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy", line=dict(color=t.SERIES_SAND, width=1.2),
                             fillcolor="rgba(198,181,142,0.35)"), row=2, col=1)
    fig.update_yaxes(title_text="EQUITY", title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text="DRAWDOWN", tickformat=".0%", title_font=dict(size=10), row=2, col=1)
    return style(fig, height)


def growth_of_100(series: dict[str, pd.Series], height: int = 230) -> go.Figure:
    """Each series rebased to 100 at its first value; the first is the highlighted one."""
    fig = go.Figure()
    palette = [t.TEAL, t.SERIES_SLATE, t.SERIES_SAND, t.NAVY, t.TEAL_BAR]
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=100 * s / s.iloc[0], name=name, line=dict(color=palette[i % len(palette)], width=1.8 if i == 0 else 1.2)))
    return style(fig, height, legend=len(series) > 1)


def monthly_returns(returns: pd.Series, height: int = 230) -> go.Figure:
    monthly = (1 + returns.dropna()).groupby([returns.dropna().index.year, returns.dropna().index.month]).prod() - 1
    labels = [f"{y}-{m:02d}" for y, m in monthly.index]
    colors = [t.SERIES_TEAL if v >= 0 else t.SERIES_SAND for v in monthly.values]
    fig = go.Figure(go.Bar(x=labels, y=monthly.values, marker_color=colors, hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(bargap=0.25)
    return style(fig, height)


def regime_allocation(regime_df: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    palette = [t.TEAL, t.SERIES_SAND, t.SERIES_SLATE, t.NAVY, t.TEAL_BAR, t.GOLD, t.SERIES_TEAL]
    for i, (label, group) in enumerate(regime_df.groupby("regime_label")):
        fig.add_trace(go.Scatter(x=group["date"], y=[label] * len(group), mode="markers", name=str(label),
                                 marker=dict(size=5, color=palette[i % len(palette)])), row=1, col=1)
    fig.add_trace(go.Scatter(x=regime_df["date"], y=regime_df["target_allocation"], name="Target allocation",
                             line=dict(color=t.TEAL, width=1.6, shape="hv")), row=2, col=1)
    fig.update_yaxes(title_text="REGIME", title_font=dict(size=10), row=1, col=1)
    fig.update_yaxes(title_text="ALLOCATION", tickformat=".0%", title_font=dict(size=10), row=2, col=1)
    return style(fig, height, legend=True)


def state_probabilities(probs: dict[str, float], height: int = 170) -> go.Figure:
    names, values = list(probs), list(probs.values())
    top = max(values) if values else 0
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=[t.TEAL if v == top else t.SERIES_TEAL for v in values],
                           hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(bargap=0.35)
    return style(fig, height)


def correlation_grid(corr: pd.DataFrame) -> str:
    """HTML grid of correlation cells; shade deepens with |r| (like the reference's 3x3 heat-grid)."""
    names = [str(c) for c in corr.columns]
    cells = ['<div class="rt-corr" style="grid-template-columns: 5.2rem repeat(%d, 1fr)">' % len(names), "<div></div>"]
    cells += [f'<div class="rt-corr-head">{esc(n)}</div>' for n in names]
    for r in corr.index:
        cells.append(f'<div class="rt-corr-head">{esc(r)}</div>')
        for c in corr.columns:
            v = float(corr.loc[r, c])
            alpha = 0.18 + 0.62 * min(abs(v), 1.0)
            cells.append(
                f'<div class="rt-corr-cell" style="background:rgba(70,152,152,{alpha:.2f})">{v:+.2f}</div>'
            )
    cells.append("</div>")
    return "".join(cells)
