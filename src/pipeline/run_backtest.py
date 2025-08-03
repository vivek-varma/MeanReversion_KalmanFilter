import polars as pl, numpy as np
from datetime import time, datetime, timezone
from indicators.kalman import kalman_beta_alpha
from indicators.ewstats import ew_z
from backtest.engine import PairBacktester, Params
from backtest.metrics import sharpe, drawdown, turnover
import matplotlib.pyplot as plt

UTC  = timezone.utc
CUT1 = datetime(2023, 1, 1, tzinfo=UTC)
CUT2 = datetime(2024, 1, 1, tzinfo=UTC)

# ---------- Load & filter to RTH (UTC 13:20–20:30) --------------------------
df = pl.read_parquet("data/processed/ZNZF_1m.parquet").sort("ts_event")

df = df.with_columns([
    pl.col("ts_event").alias("dt"),
    pl.col("ts_event").dt.time().alias("tod")
]).filter(
    (pl.col("tod") >= time(13,20)) & (pl.col("tod") <= time(20,30))
).drop("tod")

# ---------- Compute online β/α, spread, z -----------------------------------
a = df["ZN"].to_numpy()
b = df["ZF"].to_numpy()
beta, alpha = kalman_beta_alpha(a, b, q=1e-4, r=1e-2)
spread = a - beta*b - alpha
z = ew_z(spread, k=0.01)

df = df.with_columns([
    pl.Series("beta", beta),
    pl.Series("alpha", alpha),
    pl.Series("spread", spread),
    pl.Series("z", z),
])

# ---------- Define splits ----------------------------------------------------
# Train: 2018-01-02 .. 2022-12-31
# Test : 2023-01-02 .. 2023-12-31
# Valid: 2024-01-02 .. 2024-12-31
train = df.filter(pl.col("dt") < CUT1)
test  = df.filter((pl.col("dt") >= CUT1) & (pl.col("dt") < CUT2))
valid = df.filter(pl.col("dt") >= CUT2)

def run_segment(seg, params):
    arr = seg.select(["ZN","ZF","beta","alpha","z","dt"]).to_dict(as_series=False)
    bt  = PairBacktester(
            np.array(arr["ZN"]), np.array(arr["ZF"]),
            np.array(arr["beta"]), np.array(arr["alpha"]),
            np.array(arr["z"]),   np.array(arr["dt"]), params
          )
    out = bt.run()
    pnl = out["pnl"]
    ret = np.diff(pnl)
    S   = sharpe(ret)
    mdd, _ = drawdown(pnl)
    trn = turnover(out["posA"], out["posB"])
    return S, mdd, trn, pnl

# ---------- Grid search ENTRY/EXIT on train ---------------------------------
best = None
for entry in np.arange(1.5, 3.05, 0.25):
    for exit_ in np.arange(0.2, 0.55, 0.1):
        p = Params(entry_z=float(entry), exit_z=float(exit_), stop_z=5.0,
                   time_stop_bars=360, budget_usd=250_000,
                   cost_per_entry= (15.625 + 7.8125)*0.5,   # ~½ tick per leg
                   cost_per_exit = (15.625 + 7.8125)*0.5)
        S, mdd, trn, pnl = run_segment(train, p)
        score = S
        if best is None or score > best["score"]:
            best = dict(score=score, entry=entry, exit=exit_, params=p)
print("Best TRAIN:", best)

# ---------- Evaluate on TEST and VALID with fixed params --------------------
S_train, mdd_train, trn_train, pnl_train = run_segment(train, best["params"])
S_test,  mdd_test,  trn_test,  pnl_test  = run_segment(test,  best["params"])
S_val,   mdd_val,   trn_val,   pnl_val   = run_segment(valid, best["params"])

print(f"Sharpe  — train {S_train:.2f} | test {S_test:.2f} | valid {S_val:.2f}")
print(f"MaxDD $ — train {mdd_train:,.0f} | test {mdd_test:,.0f} | valid {mdd_val:,.0f}")
print(f"Trades  — train {trn_train} | test {trn_test} | valid {trn_val}")

# ---------- Plot equity curves ----------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(train["dt"], pnl_train, label="train")
plt.plot(test["dt"],  pnl_test,  label="test")
plt.plot(valid["dt"], pnl_val,   label="valid")
plt.title("Cumulative PnL ($)")
plt.legend(); plt.tight_layout(); plt.show()
