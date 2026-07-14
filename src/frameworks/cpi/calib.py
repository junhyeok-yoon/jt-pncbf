"""Split-conformal calibration of the one-sided optimism of V_hat vs V_raw.

Residual e = V_raw - V_hat on calib states within band |V_hat| <= band. eps_q(alpha) is the k-th order
statistic of e_+ = max(e, 0) (zeros included), k = ceil((n+1)*(1-alpha)); this is the standard one-sided
split-conformal quantile giving Pr[V_raw <= V_hat + eps_q(alpha)] >= 1-alpha on exchangeable data. The
opposite tail quantile of (V_hat - V_raw)_+ at the same alpha is the conservatism cost.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y, yhat, tau):
    """Mean pinball (quantile) loss at level tau; e = y - yhat. Penalizes optimism (yhat<y) at weight tau."""
    import torch
    e = y - yhat
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))


def eps_q_order_statistic(residuals_pos: np.ndarray, alpha: float) -> float:
    """k-th order statistic of the nonnegative residuals, k = ceil((n+1)*(1-alpha)). If k>n (insufficient
    calibration data for the requested coverage) return +inf (the conservative conformal convention)."""
    r = np.sort(np.asarray(residuals_pos, dtype=np.float64))
    n = r.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])                                               # 1-indexed k -> 0-indexed k-1


def calibrate(vhat: np.ndarray, vraw: np.ndarray, band: float, alphas) -> dict:
    """One-sided eps_q per alpha within |V_hat|<=band, plus the opposite-tail conservatism cost."""
    vhat = np.asarray(vhat, np.float64); vraw = np.asarray(vraw, np.float64)
    in_band = np.abs(vhat) <= band
    e = vraw[in_band] - vhat[in_band]                                    # optimism when > 0
    e_pos = np.maximum(e, 0.0)
    opp = np.maximum(-e, 0.0)                                            # (V_hat - V_raw)_+ conservatism cost
    out = {"band": band, "n_calib_band": int(in_band.sum()),
           "eps_q": {}, "opp_tail": {}}
    for a in alphas:
        out["eps_q"][str(a)] = eps_q_order_statistic(e_pos, a)
        out["opp_tail"][str(a)] = eps_q_order_statistic(opp, a)
    return out
