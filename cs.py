from typing import Tuple, List

def compute_cs_wprime(points: List[tuple]) -> Tuple[float, float]:
    """Compute Critical Speed (CS) and W' from (t_min, v_kmh) using d = CS*t + W'."""
    if len(points) < 2:
        raise ValueError("Need at least two points (t_min, v_kmh).")
    pts = sorted(points, key=lambda x: x[0])
    (t1_min, v1_kmh), (t2_min, v2_kmh) = pts[0], pts[1]
    t1, t2 = t1_min * 60, t2_min * 60
    v1, v2 = v1_kmh / 3.6, v2_kmh / 3.6
    d1, d2 = v1 * t1, v2 * t2
    CS = (d2 - d1) / (t2 - t1)
    Wp = d1 - CS * t1
    return CS, Wp

def format_cs(CS_mps: float) -> float:
    return CS_mps * 3.6

def format_wprime(W_m: float) -> float:
    return W_m
