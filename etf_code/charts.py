"""
Shared plotting utilities for Dash/Plotly charts.
Used by opti, backtest, and exposure modules to avoid duplicated figure and dcc.Graph boilerplate.
"""
import pandas as pd
import plotly.graph_objects as go
from dash import dcc

# Default height for chart iframes
DEFAULT_CHART_HEIGHT = 420

# Standard dcc.Graph config used across all charts
GRAPH_CONFIG = {"displaylogo": False, "scrollZoom": True}


def dash_graph(fig: go.Figure, height: int = DEFAULT_CHART_HEIGHT):
    """Wrap a Plotly figure in a Dash dcc.Graph with standard config and style."""
    return dcc.Graph(
        figure=fig,
        config=GRAPH_CONFIG,
        style={"height": f"{height}px"},
    )


def figure_performance_vs_benchmarks(
    portfolio_pct: pd.Series,
    spy_pct: pd.Series,
    bonds_pct: pd.Series,
    gold_pct: pd.Series,
    title: str,
) -> go.Figure:
    """
    Build a cumulative return (%) line chart: portfolio vs SPY, AGG, GLD.
    All series should be in percentage (e.g. (cumulative - 1) * 100).
    """
    hovertemplate = "%{y:.1f}%<extra>%{fullData.name}</extra>"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spy_pct.index,
            y=spy_pct.values,
            mode="lines",
            name="Stocks",
            line=dict(color="red"),
            opacity=0.6,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bonds_pct.index,
            y=bonds_pct.values,
            mode="lines",
            name="Bonds",
            line=dict(color="blue"),
            opacity=0.6,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gold_pct.index,
            y=gold_pct.values,
            mode="lines",
            name="Gold",
            line=dict(color="orange"),
            opacity=0.6,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_pct.index,
            y=portfolio_pct.values,
            mode="lines",
            name="Portfolio",
            line=dict(width=4, color="green"),
            hovertemplate=hovertemplate,
        )
    )
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        title=title,
        yaxis_title="%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_yaxes(tickformat=".1f")
    return fig


def figure_drawdown(cumulative_series: pd.Series, title: str) -> go.Figure:
    """
    Build a drawdown chart from a cumulative wealth series (e.g. (1 + returns).cumprod()).
    Drawdown is (cumulative / rolling_max - 1) * 100.
    """
    rolling_max = cumulative_series.cummax()
    drawdown = cumulative_series / rolling_max - 1
    drawdown_pct = drawdown * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="red"),
            hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_yaxes(tickformat=".1f")
    return fig
