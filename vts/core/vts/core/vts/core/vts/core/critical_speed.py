import numpy as np

def compute_critical_speed(time1_min, v1_kmh, time2_min, v2_kmh):
    v1 = v1_kmh / 3.6
    v2 = v2_kmh / 3.6
    t1 = time1_min * 60.0
    t2 = time2_min * 60.0
    CS = (v2 * t2 - v1 * t1) / (t2 - t1)
    W_prime = (v1 - CS) * t1
    return CS, W_prime

def adjust_r_by_wprime(s, r, k):
    s0, s1 = 0.3, 42.0
    if s <= s0:
        f = 1.0
    elif s >= s1:
        f = -1.0
    else:
        f = 1.0 - 2.0 * (s - s0) / (s1 - s0)
    return r * (1.0 + f * (k - 1.0))

def apply_wprime_correction(distances, r_values, k):
    return np.array([adjust_r_by_wprime(s, r, k) for s, r in zip(distances, r_values)])
