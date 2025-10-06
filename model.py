
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

def _lin_interp(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation with linear extrapolation at the ends.
    x must be strictly increasing."""
    return np.interp(xq, x, y, left=None, right=None)

def _ensure_sorted_pairs(pairs: List[Tuple[float,float]]):
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    x = np.array([p[0] for p in pairs_sorted], dtype=float)
    y = np.array([p[1] for p in pairs_sorted], dtype=float)
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-9
    return x, y

@dataclass
class IdealModel:
    s_km: np.ndarray
    t_min: np.ndarray
    v_kmh: np.ndarray

    @classmethod
    def from_distance_time_points(cls, points: List[Tuple[float, float]]):
        s, t = _ensure_sorted_pairs(points)
        v = s / (t/60.0)
        return cls(s_km=s, t_min=t, v_kmh=v)

    def v_ideal(self, s_km: np.ndarray) -> np.ndarray:
        return _lin_interp(s_km, self.s_km, self.v_kmh)

    def t_ideal(self, s_km: np.ndarray) -> np.ndarray:
        return _lin_interp(s_km, self.s_km, self.t_min)

    def time_for_distance(self, s_km: float) -> float:
        return float(self.t_ideal(np.array([s_km]))[0])

    def speed_for_distance(self, s_km: float) -> float:
        return float(self.v_ideal(np.array([s_km]))[0])

@dataclass
class PersonalModel:
    ideal: IdealModel
    user_points: List[Tuple[float, float]]
    s_r: np.ndarray
    r_of_s: np.ndarray

    @classmethod
    def from_user_points(cls, ideal: IdealModel, user_points: List[Tuple[float, float]]):
        if len(user_points) == 0:
            s_r = np.array([ideal.s_km[0], ideal.s_km[-1]], dtype=float)
            r_of_s = np.ones_like(s_r)
            return cls(ideal=ideal, user_points=[], s_r=s_r, r_of_s=r_of_s)

        s_u, v_u = _ensure_sorted_pairs(user_points)
        v_id = ideal.v_ideal(s_u)
        r = v_u / np.maximum(v_id, 1e-12)
        return cls(ideal=ideal, user_points=list(zip(s_u, v_u)), s_r=s_u, r_of_s=r)

    def r(self, s_km: np.ndarray) -> np.ndarray:
        s_clamped = np.clip(s_km, self.s_r[0], self.s_r[-1])
        return _lin_interp(s_clamped, self.s_r, self.r_of_s)

    def v_personal(self, s_km: np.ndarray) -> np.ndarray:
        return self.r(s_km) * self.ideal.v_ideal(s_km)

    def t_personal(self, s_km: np.ndarray) -> np.ndarray:
        v = np.maximum(self.v_personal(s_km), 1e-9)
        return 60.0 * s_km / v

    def time_for_distance(self, s_km: float) -> float:
        return float(self.t_personal(np.array([s_km]))[0])

    def speed_for_distance(self, s_km: float) -> float:
        return float(self.v_personal(np.array([s_km]))[0])

    def distance_for_time(self, t_min: float, s_min: float = 0.05, s_max: float = 100.0,
                          tol: float = 1e-6, iters: int = 100) -> float:
        """Find s (km) s.t. t_personal(s) == t_min via bisection."""
        lo, hi = s_min, s_max
        def f(s): return self.time_for_distance(s) - t_min
        # expand bounds
        while f(hi) < 0 and hi < 1e6:
            hi *= 2.0
        while f(lo) > 0 and lo > 1e-6:
            lo *= 0.5
        flo, fhi = f(lo), f(hi)
        if flo > 0 or fhi < 0:
            return (lo + hi) / 2.0
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            fm = f(mid)
            if abs(fm) < tol:
                return mid
            if fm > 0:
                hi = mid
            else:
                lo = mid
        return 0.5*(lo+hi)
