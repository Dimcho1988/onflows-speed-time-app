import numpy as np
from dataclasses import dataclass
from .ideal import IdealCurve

def _interp_flat(x, xp, fp):
    y = np.interp(x, xp, fp)
    y = np.where(x<xp[0], fp[0], y)
    y = np.where(x>xp[-1], fp[-1], y)
    return y

@dataclass
class PersonalizedCurve:
    ideal: IdealCurve
    s_anchor: np.ndarray  # km
    r_anchor: np.ndarray  # r = v_real / v_ideal(s)
    def r_of_s(self, s): return _interp_flat(np.asarray(s,float), self.s_anchor, self.r_anchor)
    def from_distance(self, s_km):
        v_id = float(self.ideal.v_of_s(s_km)); r = float(self.r_of_s(s_km))
        v = max(v_id*r, 1e-9); t = (s_km / v) * 60.0
        return {"distance_km": s_km, "speed_kmh": v, "time_min": t}
    def from_time(self, t_min):
        s = float(np.interp(t_min, self.ideal.t_tab, self.ideal.s_tab))
        v_id = float(self.ideal.v_of_s(s)); r = float(self.r_of_s(s))
        return {"time_min": t_min, "distance_km": s, "speed_kmh": max(v_id*r,1e-9)}
    def from_speed(self, v_kmh):
        s_grid = np.linspace(self.ideal.s_tab[0], self.ideal.s_tab[-1], 2000)
        v_id   = self.ideal.v_of_s(s_grid)
        v_p    = np.maximum(v_id * self.r_of_s(s_grid), 1e-9)
        s = float(s_grid[np.argmin(np.abs(v_p - v_kmh))])
        t = (s / v_kmh) * 60.0
        return {"speed_kmh": v_kmh, "distance_km": s, "time_min": t}
    @classmethod
    def from_sv(cls, ideal: IdealCurve, anchors):
        s, r = [], []
        for d, v_real in anchors:
            v_id = float(ideal.v_of_s(d))
            s.append(float(d)); r.append((v_real/v_id) if v_id>0 else 1.0)
        order = np.argsort(s)
        return cls(ideal, np.array(s)[order], np.array(r)[order])
