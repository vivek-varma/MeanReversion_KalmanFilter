# Backtest metrics
# ...implementation placeholder...
import numpy as np

def sharpe(ret, scale=252*6.5*12):
    m = np.nanmean(ret); s = np.nanstd(ret)
    return (m / s) * np.sqrt(scale) if s > 0 else 0.0

def drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak)
    return dd.min(), dd

def turnover(posA, posB):
    trades = np.sum((np.abs(np.diff(posA)) + np.abs(np.diff(posB))) > 0)
    return trades
