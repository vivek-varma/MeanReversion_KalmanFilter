# src/backtest/viz.py (already have plot_equity_bars)
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_equity_bars(dt, pnl, freq="D", title="Equity & PnL"):
    s = pd.Series(pnl, index=pd.DatetimeIndex(dt))
    ret = s.diff().resample(freq).sum().dropna()
    colors = np.where(ret >= 0, "rgba(34,197,94,0.8)", "rgba(239,68,68,0.8)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=ret.index, y=ret.values, marker_color=colors, name=f"{freq}-PnL", opacity=0.85)
    fig.add_scatter(x=s.index, y=s.values, name="Equity", mode="lines", line=dict(width=2), secondary_y=True)
    fig.update_layout(title=title, bargap=0, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, yanchor="bottom", xanchor="right"))
    fig.update_yaxes(title_text=f"{freq}-PnL", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Equity", secondary_y=True)
    return fig

def save_plotly(fig, path_html, path_png=None, scale=2):
    fig.write_html(str(path_html))
    if path_png is not None:
        try:
            fig.write_image(str(path_png), scale=scale)  # needs kaleido
        except Exception as e:
            print(f"[plotly] PNG export skipped ({e}). HTML saved at {path_html}")
