"""
src/chart.py
------------
Plotly chart matching the Julia reference plot:
  - scattered observed points
  - posterior predictive mean line
  - shaded uncertainty ribbon (mean ± std)
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

# dark theme — matches app.py
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#21262d"
TEXT_PRI = "#e6edf3"
TEXT_MUT = "#8b949e"
VIOLET   = "#a371f7"
SCATTER  = "#58a6ff"


def build_trend_chart(report: dict) -> Optional[go.Figure]:
    """
    Build the posterior predictive chart for one ECB report.

    Expects report["trend"] to contain the keys produced by load_and_predict():
        x_obs_orig, y_obs_orig   — scattered observed points
        x_grid_orig              — dense x grid (original units)
        y_mean, y_std            — posterior predictive mean and std
        last_obs_idx             — index of last observed point
        n_future                 — forecast horizon in days
        n_samples                — number of posterior samples
        trained_on               — date model was trained
    """
    trend = report.get("trend", {})

    # required keys
    required = ["x_obs_orig", "y_obs_orig", "x_grid_orig", "y_mean", "y_std"]
    if not all(k in trend for k in required):
        return None

    x_obs  = trend["x_obs_orig"]
    y_obs  = trend["y_obs_orig"]
    x_grid = trend["x_grid_orig"]
    y_mean = np.array(trend["y_mean"])
    y_std  = np.array(trend["y_std"])

    last_obs_idx = trend.get("last_obs_idx", len(x_obs) - 1)
    n_future     = trend.get("n_future", 30)
    n_samples    = trend.get("n_samples", 200)
    trained_on   = trend.get("trained_on", "N/A")

    title = report.get("title", "ECB Indicator")
    pair  = report.get("currency_pair", "value")

    # date labels from key_statistics
    dates         = _extract_dates(report)
    x_obs_labels  = dates if dates else list(range(len(x_obs)))
    x_grid_labels = _interpolate_labels(dates, len(x_grid)) if dates else list(range(len(x_grid)))

    # safe split index — where observed ends and forecast begins
    split = int(len(x_grid_labels) * len(x_obs) / (len(x_obs) + n_future))
    split = max(0, min(split, len(x_grid_labels) - 1))

    fig = go.Figure()

    # ── shaded ribbon: mean ± std ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(x_grid_labels) + list(x_grid_labels[::-1]),
        y=list(y_mean + y_std) + list((y_mean - y_std)[::-1]),
        fill="toself",
        fillcolor="rgba(163,113,247,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="Uncertainty (±1σ)",
        showlegend=True,
    ))

    # ── posterior mean line ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_grid_labels,
        y=y_mean.tolist(),
        mode="lines",
        name="VD posterior mean",
        line=dict(color=VIOLET, width=3),
        hovertemplate="<b>%{x}</b><br>predicted: %{y:.4f}<extra></extra>",
    ))

    # ── observed data points ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_obs_labels,
        y=y_obs,
        mode="markers",
        name=f"Observed ({pair})",
        marker=dict(color=SCATTER, size=6, line=dict(color=TEXT_PRI, width=0.5)),
        hovertemplate="<b>%{x}</b><br>observed: %{y:.4f}<extra></extra>",
    ))

    # ── dashed forecast section ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_grid_labels[split:],
        y=y_mean[split:].tolist(),
        mode="lines",
        name=f"Forecast (+{n_future}d)",
        line=dict(color=VIOLET, width=2, dash="dash"),
        hovertemplate="<b>%{x}</b><br>forecast: %{y:.4f}<extra></extra>",
    ))

    # ── forecast start vertical line ──────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_grid_labels[split], x_grid_labels[split]],
        y=[float(np.min(y_mean - y_std)), float(np.max(y_mean + y_std))],
        mode="lines",
        line=dict(color="red", width=1, dash="dot"),
        name="forecast start",
        hoverinfo="skip",
    ))

    # ── sample count annotation ───────────────────────────────
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.0, y=1.02,
        text=f"⚠ Posterior predictive based on {n_samples} samples",
        showarrow=False,
        font=dict(size=10, color=TEXT_MUT),
        align="right",
    )

    fig.update_layout(
        height = 550,
        title=dict(
            text=(
                f"{title}<br>"
                f"<sup style='color:{TEXT_MUT}'>VDM MLP [1→64→1] · sin activation · "
                f"trained on: {trained_on}</sup>"
            ),
            font=dict(size=14, color=TEXT_PRI),
            x=0,
        ),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT_MUT, size=12),
        xaxis=dict(
            title="Date (trading days)",
            showgrid=True,
            gridcolor=BORDER,
            tickangle=-45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=f"Exchange rate ({pair})",
            showgrid=True,
            gridcolor=BORDER,
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor=PANEL_BG,
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.32,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=10, r=10, t=70, b=90),
        hovermode="x unified",
    )

    return fig


# ── helpers ───────────────────────────────────────────────────

def _extract_dates(report: dict) -> list:
    # use full date list if available
    if "date_labels" in report:
        return report["date_labels"]
    # fallback: parse from key_statistics
    import re
    dates = []
    for entry in report.get("key_statistics", []):
        m = re.match(r"(\d{4}-\d{2}-\d{2}|\d{4}-\d{2}):", entry.strip())
        if m:
            dates.append(m.group(1))
    return dates


def _interpolate_labels(dates: list, n_grid: int) -> list:
    """
    Map a dense grid of n_grid points onto the date label space.
    """
    if not dates:
        return list(range(n_grid))
    indices = np.linspace(0, len(dates) - 1, n_grid)
    return [dates[int(round(i))] for i in indices]