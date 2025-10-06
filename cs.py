
from __future__ import annotations
from typing import Tuple, List

def compute_cs_wprime(time_speed_points_min_kmh: List[tuple]) -> Tuple[float, float]:
    """Compute Critical Speed (CS) and W' from (t_min, v_kmh) points using d = CS*t + W'.
    Returns (CS_mps, W_prime_m)."""
    if len(time_speed_points_min_kmh) < 2:
        raise ValueError("Need at least two (time_min, speed_kmh) points.")
    pts = sorted(time_speed_points_min_kmh, key=lambda x: x[0])
    (t1_min, v1_kmh) = pts[0]
    idx = 1
    while idx < len(pts) and abs(pts[idx][0] - t1_min) < 1e-9:
        idx += 1
    if idx >= len(pts):
        raise ValueError("Times must not be identical for both points.")
    (t2_min, v2_kmh) = pts[idx]
    t1 = t1_min * 60.0
    t2 = t2_min * 60.0
    v1 = v1_kmh * (1000.0/3600.0)
    v2 = v2_kmh * (1000.0/3600.0)
    d1 = v1 * t1
    d2 = v2 * t2
    CS = (d2 - d1) / (t2 - t1)
    W_prime = d1 - CS * t1
    return CS, W_prime

def format_cs(CS_mps: float) -> float:
    return CS_mps * 3.6

def format_wprime(W_m: float) -> float:
    return W_m
