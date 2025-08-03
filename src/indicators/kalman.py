# Kalman, z-score, ATR, etc.
# ...implementation placeholder...
import numpy as np

def kalman_beta_alpha(pa: np.ndarray, pb: np.ndarray,
                      q: float = 1e-4, r: float = 1e-2):
    """
    Online Kalman update for β, α (state x=[β, α]) with observation a_t = β b_t + α + ε.
    Returns two np.ndarray of shape (n,): beta, alpha.
    """
    n = len(pa)
    beta = np.empty(n); alpha = np.empty(n)

    # state + covariances
    x  = np.array([1.0, 0.0], dtype=float)           # initial β, α
    Q  = np.eye(2) * q
    R  = np.array([[r]])
    P  = np.eye(2)

    for t in range(n):
        b = pb[t]
        H = np.array([[b, 1.0]])                    # observation matrix
        # predict
        P = P + Q
        # innovation
        y  = pa[t] - H @ x
        S  = H @ P @ H.T + R
        K  = P @ H.T @ np.linalg.inv(S)             # 2x1
        # update
        x  = x + (K[:, 0] * y)
        P  = (np.eye(2) - K @ H) @ P

        beta[t], alpha[t] = x[0], x[1]

    return beta, alpha
