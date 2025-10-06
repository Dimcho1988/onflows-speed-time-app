import numpy as np

def compute_critical_speed(time1_min, v1_kmh, time2_min, v2_kmh):
    """
    Изчислява критична скорост (CS) и анаеробен резерв (W').
    time1_min, time2_min – време в минути
    v1_kmh, v2_kmh – скорости от индивидуалната крива (в km/h)
    """
    # Преобразуваме в m/s
    v1 = v1_kmh / 3.6
    v2 = v2_kmh / 3.6
    t1 = time1_min * 60
    t2 = time2_min * 60

    # Формули по критична скорост
    CS = (v2 * t2 - v1 * t1) / (t2 - t1)
    W_prime = (v1 - CS) * t1  # в Joules, при 1 kg (отн.)

    return CS, W_prime


def adjust_r_by_wprime(s, r, k):
    """
    Модифицира процента r(s) според анаеробния резерв.
    k = W'/W'_ideal
    f(s) = 1 за спринт, -1 за маратон (линейно между тях)
    """
    s0, s1 = 0.3, 42.0  # km
    if s <= s0:
        f = 1
    elif s >= s1:
        f = -1
    else:
        f = 1 - 2 * (s - s0) / (s1 - s0)
    return r * (1 + f * (k - 1))


def apply_wprime_correction(distances, r_values, k):
    """Прилага adjust_r_by_wprime върху цяла серия от стойности."""
    corrected = []
    for s, r in zip(distances, r_values):
        corrected.append(adjust_r_by_wprime(s, r, k))
    return np.array(corrected)
