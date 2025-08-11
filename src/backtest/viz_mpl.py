# src/backtest/viz_mpl.py
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pathlib import Path

def _fmt_dollars(y, _pos=None):
    ay = abs(y)
    if ay >= 1e9:  return f"${y/1e9:.1f}B"
    if ay >= 1e6:  return f"${y/1e6:.1f}M"
    if ay >= 1e3:  return f"${y/1e3:.0f}K"
    return f"${y:.0f}"

def _bar_width_from_index(idx):
    x = mdates.date2num(pd.DatetimeIndex(idx).to_pydatetime())
    return 0.8 * float(np.median(np.diff(x))) if len(x) > 1 else 0.8

def save_equity_bars_png(dt, pnl, out_path, freq="D", title="Equity & PnL"):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    s = pd.Series(pnl, index=pd.DatetimeIndex(dt))
    ret = s.diff().resample(freq).sum().dropna()
    colors = np.where(ret.values >= 0, "#22c55e", "#ef4444")  # green/red

    fig, ax1 = plt.subplots(figsize=(11, 4))
    x = mdates.date2num(ret.index.to_pydatetime())
    ax1.bar(x, ret.values, width=_bar_width_from_index(ret.index), color=colors, alpha=0.85)
    ax1.xaxis_date()
    ax1.set_ylabel(f"{freq} PnL")
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))
    #ax1.ticklabel_format(axis="y", style="plain")  # disable 1eN

    ax2 = ax1.twinx()
    ax2.plot(s.index, s.values, lw=1.8, color="#1f2937")
    ax2.set_ylabel("Equity")
    ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))
    #ax2.ticklabel_format(axis="y", style="plain")

    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_equity_combined_png(train_dt, train_pnl, test_dt, test_pnl, valid_dt, valid_pnl, out_path):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.DatetimeIndex(train_dt), train_pnl, label="train", lw=1.8)
    ax.plot(pd.DatetimeIndex(test_dt),  test_pnl,  label="test",  lw=1.8)
    ax.plot(pd.DatetimeIndex(valid_dt), valid_pnl, label="valid", lw=1.8)
    ax.set_title("Cumulative PnL ($) — Train/Test/Validation")
    ax.set_ylabel("Equity ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dollars))
    #ax.ticklabel_format(axis="y", style="plain")
    ax.grid(True, alpha=0.25); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
