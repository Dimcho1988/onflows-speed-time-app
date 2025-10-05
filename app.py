import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
#  Helpers: parsing & units
# -----------------------------
def parse_time_to_min(s: str) -> float:
    s = s.strip().lower().replace(" ", "")
    if not s:
        return math.nan
    # plain number -> minutes
    if s.replace(".", "", 1).isdigit():
        return float(s)
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = parts
            return float(m) + float(sec)/60
        if len(parts) == 3:
            h, m, sec = parts
            return 60*float(h) + float(m) + float(sec)/60
    # tokens like 1h28m30s or 40s
    import re
    h = re.search(r"(\d+(?:\.\d+)?)h", s)
    m = re.search(r"(\d+(?:\.\d+)?)m", s)
    sec = re.search(r"(\d+(?:\.\d+)?)s", s)
    total = 0.0
    if h: total += 60*float(h.group(1))
    if m: total += float(m.group(1))
    if sec: total += float(sec.group(1))/60
    if total > 0:
        return total
    # last fallback 'Xs' seconds only
    if s.endswith("s") and s[:-1].replace(".", "", 1).isdigit():
        return float(s[:-1]) / 60
    return math.nan

def parse_speed_to_kmh(s: str) -> float:
    s = s.strip().lower().replace(" ", "")
    import re
    # pace formats
    if "min/km" in s or "/km" in s:
        mmss = re.findall(r"(\d+):(\d+)", s)
        if mmss:
            mm, ss = mmss[0]
            pace_min = float(mm) + float(ss)/60
            return 60.0/pace_min
        m = re.search(r"(\d+(?:\.\d+)?)min/km", s)
        if m:
            return 60.0/float(m.group(1))
    # numeric → km/h
    s_num = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    return float(s_num) if s_num else math.nan

def parse_distance_to_km(s) -> float:
    # приемаме и числа, и текст
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower().replace(" ", "")
    if not s:
        return math.nan
    if s in {"marathon", "maraton"}:
        return 42.195
    if s in {"half", "halfmarathon", "polumaraton"}:
        return 21.097
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    if s.endswith("km"):
        return float(s[:-2])
    if s.endswith("k"):
        return float(s[:-1])
    if s.replace(".", "", 1).isdigit():
        return float(s)
    return math.nan

# -----------------------------
#  Monotone cubic interpolator (Fritsch–Carlson)
# -----------------------------
@dataclass
class MonotoneCubic:
    x: np.ndarray
    y: np.ndarray
    def __post_init__(self):
        x = np.asarray(self.x, float); y = np.asarray(self.y, float)
        if np.any(np.diff(x) <= 0): raise ValueError("x must be strictly increasing")
        self.x, self.y = x, y
        h = np.diff(x); delta = np.diff(y)/h
        m = np.zeros_like(y)
        m[0] = delta[0]; m[-1] = delta[-1]
        for i in range(1, len(x)-1):
            if delta[i-1]*delta[i] <= 0:
                m[i] = 0.0
            else:
                w1 = 2*h[i] + h[i-1]; w2 = h[i] + 2*h[i-1]
                m[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])
        self.h, self.delta, self.m = h, delta, m
    def _idx(self, xv: float) -> int:
        if xv <= self.x[0]: return 0
        if xv >= self.x[-1]: return len(self.x)-2
        return int(np.searchsorted(self.x, xv) - 1)
    def __call__(self, xq):
        xq = np.atleast_1d(np.asarray(xq, float)); out = np.empty_like(xq)
        for j, xv in enumerate(xq):
            if xv < self.x[0]: out[j] = self.y[0] + (xv-self.x[0])*self.delta[0]; continue
            if xv > self.x[-1]: out[j] = self.y[-1] + (xv-self.x[-1])*self.delta[-1]; continue
            i = self._idx(xv); h = self.h[i]; t = (xv-self.x[i])/h
            y0,y1 = self.y[i], self.y[i+1]; m0,m1 = self.m[i], self.m[i+1]
            t2=t*t; t3=t2*t
            h00=(2*t3-3*t2+1); h10=(t3-2*t2+t); h01=(-2*t3+3*t2); h11=(t3-t2)
            out[j] = h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1
        return out if out.shape != () else out.item()
    def invert_monotone(self, yq: float, lo=None, hi=None, tol=1e-8, maxit=50) -> float:
        if lo is None: lo = self.x[0]
        if hi is None: hi = self.x[-1]
        f_lo = self(lo)-yq; f_hi = self(hi)-yq
        if f_lo*f_hi > 0:
            span = hi-lo
            for _ in range(3):
                lo2 = lo-0.2*span; hi2 = hi+0.2*span
                f_lo = self(lo2)-yq; f_hi = self(hi2)-yq
                lo,hi = lo2,hi2; span = hi-lo
                if f_lo*f_hi <= 0: break
        for _ in range(maxit):
            mid = 0.5*(lo+hi); f_mid = self(mid)-yq
            if abs(f_mid) < tol or (hi-lo) < tol: return mid
            if f_lo*f_mid <= 0: hi = mid; f_hi = f_mid
            else: lo = mid; f_lo = f_mid
        return 0.5*(lo+hi)

# -----------------------------
#  Ideal model (v(s), t(s))
# -----------------------------
class DistanceTimeSpeedModel:
    def __init__(self, distance_km: np.ndarray, time_min: np.ndarray):
        s = np.asarray(distance_km, float); t_h = np.asarray(time_min, float)/60.0
        ord_ = np.argsort(s); s = s[ord_]; t_h = t_h[ord_]
        v = s/t_h
        self.v_of_s = MonotoneCubic(s, v)
        self.t_of_s = MonotoneCubic(s, t_h)
    def from_distance(self, s_km: float) -> dict:
        v = float(self.v_of_s(s_km)); t = float(self.t_of_s(s_km))
        return {"distance_km": s_km, "speed_kmh": v, "time_h": t, "time_min": t*60}
    def from_time(self, t_min: float) -> dict:
        T_h = t_min/60.0; s = float(self.t_of_s.invert_monotone(T_h)); v = float(self.v_of_s(s))
        return {"time_min": t_min, "time_h": T_h, "distance_km": s, "speed_kmh": v}
    def from_speed(self, v_kmh: float) -> dict:
        s = float(self.v_of_s.invert_monotone(v_kmh)); t = float(self.t_of_s(s))
        return {"speed_kmh": v_kmh, "distance_km": s, "time_h": t, "time_min": t*60}

# -----------------------------
#  Personalized model: v_pers(s) = r(s) * v_ideal(s)
#  r(s) – линейно между анкърите, плоско извън тях
# -----------------------------
def piecewise_flat_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    y = np.empty_like(x, float)
    y[x <= xp[0]] = fp[0]; y[x >= xp[-1]] = fp[-1]
    m = (x > xp[0]) & (x < xp[-1])
    if np.any(m): y[m] = np.interp(x[m], xp, fp)
    return y

@dataclass
class PersonalizedModel:
    ideal: DistanceTimeSpeedModel
    s_grid: np.ndarray
    v_pers: np.ndarray
    t_pers: np.ndarray
    @classmethod
    def from_anchors_sv(cls, ideal: DistanceTimeSpeedModel, anchors: Iterable[Tuple[float,float]], ds:float=0.01):
        # anchors: (distance_km, speed_kmh)
        s_a, r_a = [], []
        for s, v_real in anchors:
            v_id = float(ideal.v_of_s(s)); r = (v_real/v_id) if v_id>0 else 1.0
            s_a.append(float(s)); r_a.append(float(r))
        return cls._build(ideal, np.array(s_a), np.array(r_a), ds)
    @classmethod
    def from_anchors_ts(cls, ideal: DistanceTimeSpeedModel, anchors: Iterable[Tuple[float,float]], ds:float=0.01):
        # anchors: (time_min, speed_kmh)
        s_a, r_a = [], []
        for t_min, v_real in anchors:
            out = ideal.from_time(t_min); s = out["distance_km"]; v_id = out["speed_kmh"]
            r = (v_real/v_id) if v_id>0 else 1.0
            s_a.append(s); r_a.append(r)
        return cls._build(ideal, np.array(s_a), np.array(r_a), ds)
    @classmethod
    def from_anchors_td(cls, ideal: DistanceTimeSpeedModel, anchors: Iterable[Tuple[float,float]], ds:float=0.01):
        # anchors: (time_min, distance_km)
        s_a, r_a = [], []
        for t_min, d_real in anchors:
            v_real = d_real/(t_min/60.0)
            out = ideal.from_time(t_min); s = out["distance_km"]; v_id = out["speed_kmh"]
            r = (v_real/v_id) if v_id>0 else 1.0
            s_a.append(s); r_a.append(r)
        return cls._build(ideal, np.array(s_a), np.array(r_a), ds)
    @classmethod
    def _build(cls, ideal, s_anchor, r_anchor, ds):
        ord_ = np.argsort(s_anchor); s_anchor = s_anchor[ord_]; r_anchor = r_anchor[ord_]
        uniq = np.concatenate(([True], np.diff(s_anchor) > 1e-9))
        s_anchor = s_anchor[uniq]; r_anchor = r_anchor[uniq]
        s_min = float(ideal.v_of_s.x[0]); s_max = float(ideal.v_of_s.x[-1])
        s_grid = np.arange(max(1e-6, s_min), s_max+ds, ds, float)
        r_grid = piecewise_flat_interp(s_grid, s_anchor, r_anchor)
        v_id = ideal.v_of_s(s_grid)
        v_pers = np.maximum(r_grid * v_id, 1e-6)
        inv_v = 1.0/v_pers
        t_pers = np.zeros_like(s_grid); t_pers[1:] = np.cumsum(0.5*(inv_v[1:]+inv_v[:-1])*np.diff(s_grid))
        return cls(ideal=ideal, s_grid=s_grid, v_pers=v_pers, t_pers=t_pers)
    # queries
    def from_distance(self, s_km: float) -> dict:
        v = float(np.interp(s_km, self.s_grid, self.v_pers))
        t_h = float(np.interp(s_km, self.s_grid, self.t_pers))
        return {"distance_km": s_km, "speed_kmh": v, "time_min": t_h*60, "time_h": t_h}
    def from_time(self, t_min: float) -> dict:
        T_h = t_min/60.0
        s = float(np.interp(T_h, self.t_pers, self.s_grid))
        v = float(np.interp(s, self.s_grid, self.v_pers))
        return {"distance_km": s, "speed_kmh": v, "time_min": t_min, "time_h": T_h}
    def from_speed(self, v_kmh: float) -> dict:
        idx = int(np.argmin(np.abs(self.v_pers - v_kmh)))
        s = float(self.s_grid[idx]); t_h = float(self.t_pers[idx])
        return {"distance_km": s, "speed_kmh": v_kmh, "time_min": t_h*60, "time_h": t_h}

# -----------------------------
#  Load ideal curve
# -----------------------------
@st.cache_data
def load_ideal(csv_file) -> DistanceTimeSpeedModel:
    df = pd.read_csv(csv_file)
    return DistanceTimeSpeedModel(df["distance_km"].values, df["time_min"].values)

# -----------------------------
#  UI
# -----------------------------
st.set_page_config(page_title="VTS Predictor", layout="wide")
st.title("VTS Predictor — Ideal vs Personalized")

# Data source
col1, col2 = st.columns([2,1])
with col1:
    st.write("1) Зареди **идеалната крива** (CSV с колони `distance_km`, `time_min`). По подразбиране използвай `ideal_distance_time_speed.csv` в папката.")
    uploaded = st.file_uploader("По желание качи CSV, за да замениш идеалната крива", type=["csv"])
with col2:
    st.info("Може да оставиш полето празно — ако файлът е в същата папка, приложението ще го зареди.")

# Load ideal
default_csv_path = "ideal_distance_time_speed.csv"
try:
    ideal = load_ideal(uploaded if uploaded is not None else default_csv_path)
except Exception as e:
    st.error(f"Грешка при зареждане на идеалната крива: {e}")
    st.stop()

# Sidebar: enter real points (anchors)
st.sidebar.header("Персонализация (опорни точки)")
mode = st.sidebar.radio("Тип точки", ["distance + speed", "time + speed", "time + distance"])

anchors: List[Tuple[float,float]] = []
n_points = st.sidebar.number_input("Брой точки", 1, 20, value=3)
for i in range(n_points):
    if mode == "distance + speed":
        d = st.sidebar.text_input(f"[{i+1}] distance (km, k, m)", value=["2km","13km","21.1km"][i] if i<3 else "")
        v = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", value=["16","13","21"][i] if i<3 else "")
        dk = parse_distance_to_km(d); vk = parse_speed_to_kmh(v)
        if not math.isnan(dk) and not math.isnan(vk): anchors.append((dk, vk))
    elif mode == "time + speed":
        t = st.sidebar.text_input(f"[{i+1}] time (min, 1:30, 40s, 1h28m)", value=["7","67","88"][i] if i<3 else "")
        v = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", value=["16","18","20"][i] if i<3 else "")
        tm = parse_time_to_min(t); vk = parse_speed_to_kmh(v)
        if not math.isnan(tm) and not math.isnan(vk): anchors.append((tm, vk))
    else:
        t = st.sidebar.text_input(f"[{i+1}] time (min, 1:30, 40s, 1h28m)", value=["30","60","120"][i] if i<3 else "")
        d = st.sidebar.text_input(f"[{i+1}] distance (km, k, m)", value=["5km","10km","21.1km"][i] if i<3 else "")
        tm = parse_time_to_min(t); dk = parse_distance_to_km(d)
        if not math.isnan(tm) and not math.isnan(dk): anchors.append((tm, dk))

st.sidebar.caption("Остави празно за неизползвани точки.")

# Build personalized model
pm = None
if anchors:
    try:
        if mode == "distance + speed":
            pm = PersonalizedModel.from_anchors_sv(ideal, anchors, ds=0.01)
        elif mode == "time + speed":
            pm = PersonalizedModel.from_anchors_ts(ideal, anchors, ds=0.01)
        else:
            pm = PersonalizedModel.from_anchors_td(ideal, anchors, ds=0.01)
    except Exception as e:
        st.error(f"Грешка при персонализация: {e}")

# Query panel
st.header("Бързи заявки")
qc1, qc2, qc3 = st.columns(3)
with qc1:
    qd = st.text_input("Distance (km, k, m)", "1km")
    if st.button("→ от дистанция"):
        s = parse_distance_to_km(qd)
        target = pm if pm is not None else ideal
        out = target.from_distance(s)
        st.write(out)

with qc2:
    qt = st.text_input("Time (min, 1:30, 40s, 1h28m)", "88")
    if st.button("→ от време"):
        t_min = parse_time_to_min(qt)
        target = pm if pm is not None else ideal
        out = target.from_time(t_min)
        st.write(out)

with qc3:
    qv = st.text_input("Speed (km/h или 5:00 min/km)", "22")
    if st.button("→ от скорост"):
        v_kmh = parse_speed_to_kmh(qv)
        target = pm if pm is not None else ideal
        out = target.from_speed(v_kmh)
        st.write(out)

# Range table & deviation
st.header("Таблица: 100 m → Маратон + отклонение от идеала")
standard_d = [0.1,0.2,0.3,0.4,0.5,0.8,1.0,1.5,2.0,3.0,5.0,8.0,10.0,12.0,15.0,16.09,20.0,21.097,25.0,30.0,32.18,35.0,40.0,42.195]
target = pm if pm is not None else ideal
rows=[]
for s in standard_d:
    id_out = ideal.from_distance(s)
    pr_out = target.from_distance(s)
    pct_speed = 100.0 * pr_out["speed_kmh"]/id_out["speed_kmh"]
    pct_time  = 100.0 * pr_out["time_min"]/id_out["time_min"]
    rows.append({
        "distance_km": s,
        "ideal_speed_kmh": id_out["speed_kmh"], "personal_speed_kmh": pr_out["speed_kmh"], "speed_%_of_ideal": pct_speed,
        "ideal_time_min": id_out["time_min"],   "personal_time_min": pr_out["time_min"],   "time_%_of_ideal": pct_time
    })
table = pd.DataFrame(rows)
st.dataframe(table, use_container_width=True)

# Plot deviation
st.subheader("Отклонение от идеала (скорост, %)")
st.line_chart(table.set_index("distance_km")["speed_%_of_ideal"])
