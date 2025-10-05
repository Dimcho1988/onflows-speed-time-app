import numpy as np
from dataclasses import dataclass

def _lin_extrap(x, xp, fp):
    y = np.interp(x, xp, fp)
    m0 = (fp[1]-fp[0])/(xp[1]-xp[0])
    m1 = (fp[-1]-fp[-2])/(xp[-1]-xp[-2])
    y = np.where(x<xp[0],  fp[0]  + (x-xp[0]) * m0, y)
    y = np.where(x>xp[-1], fp[-1] + (x-xp[-1]) * m1, y)
    return y

@dataclass(frozen=True)
class IdealCurve:
    s_tab: np.ndarray  # km
    t_tab: np.ndarray  # min
    def v_tab(self): return self.s_tab / (self.t_tab/60.0)  # km/h
    def v_of_s(self, s): return _lin_extrap(np.asarray(s,float), self.s_tab, self.v_tab())
    def t_of_s(self, s): return _lin_extrap(np.asarray(s,float), self.s_tab, self.t_tab)
    def from_distance(self, s_km):
        v = float(self.v_of_s(s_km)); t = float(self.t_of_s(s_km))
        return {"distance_km": s_km, "speed_kmh": v, "time_min": t}
    def from_time(self, t_min):
        s = float(np.interp(t_min, self.t_tab, self.s_tab))
        v = float(self.v_of_s(s));  return {"time_min": t_min, "distance_km": s, "speed_kmh": v}
    def from_speed(self, v_kmh):
        s_grid = np.linspace(self.s_tab[0], self.s_tab[-1], 2000)
        v_grid = self.v_of_s(s_grid)
        s = float(s_grid[np.argmin(np.abs(v_grid - v_kmh))])
        t = float(self.t_of_s(s))
        return {"speed_kmh": v_kmh, "distance_km": s, "time_min": t}
