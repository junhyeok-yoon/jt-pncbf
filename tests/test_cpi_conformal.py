"""CPI conformal: order-statistic index correctness for eps_q, including ties and small n."""
from __future__ import annotations

import math

import numpy as np

from src.frameworks.cpi.calib import eps_q_order_statistic


def test_eps_q_basic_index():
    r = np.arange(100, dtype=float)                   # 0..99, n=100
    # alpha=0.10 -> k=ceil(101*0.90)=91 -> r[90]=90
    assert eps_q_order_statistic(r, 0.10) == 90.0
    # alpha=0.05 -> k=ceil(101*0.95)=96 -> r[95]=95
    assert eps_q_order_statistic(r, 0.05) == 95.0
    # alpha=0.01 -> k=ceil(101*0.99)=100 -> r[99]=99
    assert eps_q_order_statistic(r, 0.01) == 99.0


def test_eps_q_ties():
    r = np.array([0.0] * 50 + [1.0] * 50)             # n=100, k=91 -> sorted[90]=1.0
    assert eps_q_order_statistic(r, 0.10) == 1.0


def test_eps_q_small_n_returns_inf():
    # n=1, alpha=0.10 -> k=ceil(2*0.9)=2 > 1 -> inf
    assert math.isinf(eps_q_order_statistic(np.array([5.0]), 0.10))
    # n=18, alpha=0.05 -> k=ceil(19*0.95)=ceil(18.05)=19 > 18 -> inf
    assert math.isinf(eps_q_order_statistic(np.arange(18, dtype=float), 0.05))
    # n=19, alpha=0.05 -> k=ceil(20*0.95)=19 <= 19 -> finite (sorted[18])
    assert eps_q_order_statistic(np.arange(19, dtype=float), 0.05) == 18.0


def test_eps_q_empty():
    assert math.isinf(eps_q_order_statistic(np.array([]), 0.10))
