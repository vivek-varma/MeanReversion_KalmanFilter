# Minimal event loop + broker
# ...implementation placeholder...
import numpy as np
from dataclasses import dataclass

# === Contract economics (points → $) =========================================
POINT_VALUE_USD = 1000.0         # $ per full point for ZN/ZF
TICK_ZN = 1/64                    # 0.5/32
TICK_ZF = 1/128                   # 0.25/32
TICKVAL_ZN = POINT_VALUE_USD * TICK_ZN   # $15.625
TICKVAL_ZF = POINT_VALUE_USD * TICK_ZF   # $7.8125

@dataclass
class Params:
    entry_z: float = 2.0
    exit_z:  float = 0.3
    stop_z:  float = 5.0
    time_stop_bars: int = 360            # 6 hours at 1m
    budget_usd: float = 250_000.0
    cost_per_entry: float = 0.0          # $ per spread entry
    cost_per_exit:  float = 0.0          # $ per spread exit

class PairBacktester:
    """
    Trade ZN vs ZF using spread = a - β b - α.
    At entry we fix quantities (qA, qB) from β_t and scale to 'budget_usd'.
    PnL in dollars: qA*Δa*1000 + qB*Δb*1000.
    """
    def __init__(self, a, b, beta, alpha, z, ts, params: Params):
        self.a = a.astype(float)
        self.b = b.astype(float)
        self.beta  = beta.astype(float)
        self.alpha = alpha.astype(float)
        self.z = z.astype(float)
        self.ts = ts
        self.p = params

    def _size(self, i, sign):
        """
        Compute integer contract quantities at index i for a single 'spread unit',
        scaled to 'budget_usd'.
        """
        pa, pb, beta = self.a[i], self.b[i], self.beta[i]
        # one spread unit: qA= +1*sign, qB= -beta*sign (β dimensionless)
        qA = 1 * sign
        qB = -beta * sign

        # notional per unit in $:
        notional = abs(qA)*pa*POINT_VALUE_USD + abs(qB)*pb*POINT_VALUE_USD
        scale = max(1.0, self.p.budget_usd / notional)
        # integer sizes with sign, at least 1 on each side
        qA = int(np.sign(qA) * max(1, int(round(abs(qA) * scale))))
        qB = int(np.sign(qB) * max(1, int(round(abs(qB) * scale))))
        return qA, qB

    def run(self):
        n = len(self.a)
        pnl = np.zeros(n)
        posA = np.zeros(n, dtype=int)  # contracts
        posB = np.zeros(n, dtype=int)
        age  = 0

        in_pos = False
        qA = qB = 0

        for i in range(1, n):
            # 1) Book PnL with positions held on (i-1 -> i)
            if in_pos:
                pnl[i] = pnl[i-1] + qA*(self.a[i]-self.a[i-1])*POINT_VALUE_USD \
                                   + qB*(self.b[i]-self.b[i-1])*POINT_VALUE_USD
                age += 1
            else:
                pnl[i] = pnl[i-1]

            posA[i], posB[i] = qA, qB

            # 2) Decide what to hold *next* bar using info at close of bar i
            z = self.z[i]
            if in_pos:
                if np.isnan(z) or abs(z) < self.p.exit_z or abs(z) > self.p.stop_z or age >= self.p.time_stop_bars:
                    in_pos = False
                    pnl[i] -= (self.p.cost_per_exit)       # exit cost
                    qA = qB = 0
                    age = 0

            if not in_pos and not np.isnan(z) and abs(z) > self.p.entry_z:
                # enter for next bar
                sgn = -np.sign(z)
                qA, qB = self._size(i, sgn)
                in_pos = True
                pnl[i] -= (self.p.cost_per_entry)         # entry cost

        return {
            "pnl": pnl,
            "posA": posA,
            "posB": posB,
        }
