import streamlit as st
import pandas as pd
import numpy as np
import math

# -------------------- Page config --------------------
st.set_page_config(page_title="VTS Predictor — Ideal vs Personalized", layout="wide")
st.title("VTS Predictor — Ideal vs Personalized")

# -------------------- Helpers --------------------
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
    # ВАЖНО: първо km/k, после m
    if s.endswith("km"):
        return float(s[:-2])
    if s.endswith("k"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    if s.replace(".", "", 1).isdigit():
        return float(s)
    return math.nan

def parse_speed_to_kmh(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower().replace(" ", "")
    if not s:
        return math.nan
    # темпо min/km
    if "min/km" in s or "/km" in s:
        parts = [p for p in s.split("/") if "min" in p or ":" in p]  # tolerant
        mm_ss = s.split("min")[0]
        if ":" in mm_ss:
            mm, ss = mm_ss.split(":")
            pace_min = float(mm) + float(ss)/60.0
            return 60.0/pace_min
        try:
            pace_min = float(mm_ss)
            return 60.0/pace_min
        except:
            pass
    # чисто число (приемаме km/h)
    num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    return float(num) if num else math.nan

def format_minutes_to_hms(minutes: float) -> str:
    if pd.isna(minutes) or minutes <= 0:
        return "-"
    total_seconds = int(round(minutes * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def lin_interp_extrap(x, xp, fp):
    """Линейна интерполация + линейна екстраполация в краищата (без SciPy)."""
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    y = np.interp(x, xp, fp)  # вътрешно
    # ляв край
    m0 = (fp[1] - fp[0]) / (xp[1] - xp[0])
    y = np.where(x < xp[0], fp[0] + (x - xp[0]) * m0, y)
    # десен край
    m1 = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    y = np.where(x > xp[-1], fp[-1] + (x - xp[-1]) * m1, y)
    return y

def interp_flat_ends(x, xp, fp):
    """Линейна интерполация; извън краищата държи постоянна стойност."""
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    y = np.interp(x, xp, fp)
    y = np.where(x < xp[0], fp[0], y)
    y = np.where(x > xp[-1], fp[-1], y)
    return y

# -------------------- Load ideal curve --------------------
st.markdown(
    "1) Зареди **идеалната крива** (CSV с колони `distance_km`, `time_min`). "
    "Можеш да качиш файл или да оставиш празно, за да се използва `ideal_distance_time_speed.csv` от репото."
)
uploaded = st.file_uploader("Качи CSV (по желание)", type=["csv"])

try:
    df_ideal = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv("ideal_distance_time_speed.csv")
except Exception as e:
    st.error(f"Грешка при зареждане на CSV: {e}")
    st.stop()

# почисти/сортирай
df_ideal = df_ideal.dropna(subset=["distance_km", "time_min"]).copy()
df_ideal = df_ideal.sort_values("distance_km")
s_tab = df_ideal["distance_km"].values.astype(float)
t_tab = df_ideal["time_min"].values.astype(float)
v_tab = s_tab / (t_tab / 60.0)  # km / (min/60) = km/h

# помощни „идеални“ функции
def ideal_from_distance(s):
    v = float(lin_interp_extrap(s, s_tab, v_tab))
    t = float(lin_interp_extrap(s, s_tab, t_tab))
    return v, t  # km/h, min

def ideal_from_time(t_min):
    # инверсия на t(s) чрез np.interp (монотонно растящо)
    s = float(np.interp(t_min, t_tab, s_tab))
    v = float(lin_interp_extrap(s, s_tab, v_tab))
    return s, v

# -------------------- Sidebar: anchors --------------------
st.sidebar.header("Персонализация (опорни точки)")
mode = st.sidebar.radio("Тип точки", ["distance + speed", "time + speed", "time + distance"])
n_pts = st.sidebar.number_input("Брой точки", min_value=1, max_value=10, value=3)

anchors_s = []   # в домейна на дистанцията
anchors_ratio = []  # r = v_real / v_ideal(s)

for i in range(n_pts):
    st.sidebar.markdown(f"**[{i+1}] точка**")
    if mode == "distance + speed":
        d_txt = st.sidebar.text_input(f"[{i+1}] distance (km/k/m)", value="" if i>=2 else ["2km","13km",""][i])
        v_txt = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", value="" if i>=2 else ["16","13",""][i])
        if d_txt and v_txt:
            s = parse_distance_to_km(d_txt); v_real = parse_speed_to_kmh(v_txt)
            if not math.isnan(s) and not math.isnan(v_real):
                v_id, _t_id = ideal_from_distance(s)
                anchors_s.append(s); anchors_ratio.append(v_real / v_id if v_id>0 else 1.0)
    elif mode == "time + speed":
        t_val = st.sidebar.text_input(f"[{i+1}] time (мин или mm:ss)", value="" if i>=2 else ["7","67",""][i])
        v_txt = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", value="" if i>=2 else ["16","18",""][i])
        if t_val and v_txt:
            # позволи mm:ss
            if ":" in t_val:
                mm, ss = t_val.split(":"); t_min = float(mm) + float(ss)/60.0
            else:
                t_min = float(t_val)
            s, v_id = ideal_from_time(t_min)
            v_real = parse_speed_to_kmh(v_txt)
            if not math.isnan(v_real):
                anchors_s.append(s); anchors_ratio.append(v_real / v_id if v_id>0 else 1.0)
    else:  # time + distance
        t_val = st.sidebar.text_input(f"[{i+1}] time (мин или mm:ss)", value="" if i>=2 else ["30","60",""][i])
        d_txt = st.sidebar.text_input(f"[{i+1}] distance (km/k/m)", value="" if i>=2 else ["5km","10km",""][i])
        if t_val and d_txt:
            if ":" in t_val:
                mm, ss = t_val.split(":"); t_min = float(mm) + float(ss)/60.0
            else:
                t_min = float(t_val)
            s = parse_distance_to_km(d_txt)
            if not math.isnan(s):
                v_real = s / (t_min/60.0) if t_min>0 else 0.0
                _s_id, v_id = ideal_from_time(t_min)
                anchors_s.append(_s_id); anchors_ratio.append(v_real / v_id if v_id>0 else 1.0)

# r(s) интерполация (линейна, плоски краища)
r_grid_fn = None
if len(anchors_s) >= 1:
    order = np.argsort(anchors_s)
    s_anchor = np.asarray(anchors_s, float)[order]
    r_anchor = np.asarray(anchors_ratio, float)[order]
    def r_of_s(x):
        return interp_flat_ends(x, s_anchor, r_anchor)
else:
    def r_of_s(x):
        return np.ones_like(np.asarray(x, float))

# -------------------- Range table --------------------
st.header("Таблица: 100 m → Маратон + отклонение от идеала")
distances = np.array(
    [0.1,0.2,0.3,0.4,0.5,0.8,1.0,1.5,2.0,3.0,5.0,8.0,10.0,12.0,15.0,16.09,20.0,21.097,25.0,30.0,32.18,35.0,40.0,42.195],
    dtype=float
)

rows = []
for s in distances:
    v_id, t_id = ideal_from_distance(s)
    ratio = float(r_of_s(s))
    v_prs = max(v_id * ratio, 1e-9)
    t_prs = (s / v_prs) * 60.0
    rows.append({
        "Дистанция (km)": s,
        "Идеална скорост (km/h)": v_id,
        "Персонална скорост (km/h)": v_prs,
        "Идеално време (ч:мм:сс)": format_minutes_to_hms(t_id),
        "Персонално време (ч:мм:сс)": format_minutes_to_hms(t_prs),
        "% от идеала (скорост)": 100.0 * v_prs / v_id if v_id>0 else np.nan
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# Export
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Изтегли резултатите (CSV)", data=csv_bytes, file_name="vts_results.csv", mime="text/csv")
# ===================== ADDON: Deviation chart + Quick queries =====================

st.subheader("📈 Отклонение от идеала (скорост, %)")
# чертаем линия по вече създадената таблица df
try:
    dev_series = df.set_index("Дистанция (km)")["% от идеала (скорост)"]
    st.line_chart(dev_series)
except Exception as _e:
    st.info("Добави поне една опорна точка, за да има отклонение за визуализация.")

st.subheader("⚡ Бързи заявки (от дистанция / време / скорост)")

# малък помощник за парс на време mm:ss -> минути
def _parse_time_to_min(x: str) -> float:
    x = str(x).strip()
    if not x:
        return float("nan")
    if ":" in x:
        mm, ss = x.split(":")
        return float(mm) + float(ss)/60.0
    return float(x)

# --- заявки ---
qc1, qc2, qc3 = st.columns(3)

with qc1:
    st.markdown("**От дистанция → време и скорост**")
    d_in = st.text_input("Дистанция (km/k/m)", key="q_dist", value="1km")
    if st.button("Изчисли от дистанция"):
        s = parse_distance_to_km(d_in)
        if math.isnan(s):
            st.error("Невалидна дистанция.")
        else:
            v_id, t_id = ideal_from_distance(s)
            r = float(r_of_s(s))
            v_p = max(v_id * r, 1e-9)
            t_p = (s / v_p) * 60.0
            st.write(f"**Персонална скорост:** {v_p:.3f} km/h")
            st.write(f"**Персонално време:** {format_minutes_to_hms(t_p)}")
            st.caption(f"(Идеал: {v_id:.3f} km/h, {format_minutes_to_hms(t_id)})")

with qc2:
    st.markdown("**От време → дистанция и скорост**")
    t_in = st.text_input("Време (мин или mm:ss)", key="q_time", value="30")
    if st.button("Изчисли от време"):
        t_min = _parse_time_to_min(t_in)
        if math.isnan(t_min) or t_min <= 0:
            st.error("Невалидно време.")
        else:
            # инверсия на t(s): s по идеала
            s_id = float(np.interp(t_min, t_tab, s_tab))
            v_id = float(lin_interp_extrap(s_id, s_tab, v_tab))
            r = float(r_of_s(s_id))
            v_p = max(v_id * r, 1e-9)
            st.write(f"**Персонална скорост:** {v_p:.3f} km/h")
            st.write(f"**Персонална дистанция:** {s_id:.3f} km")
            st.caption(f"(Идеал: {v_id:.3f} km/h, дистанция {s_id:.3f} km)")

with qc3:
    st.markdown("**От скорост → дистанция и време**")
    v_in = st.text_input("Скорост (km/h или 5:00 min/km)", key="q_speed", value="22")
    if st.button("Изчисли от скорост"):
        v_target = parse_speed_to_kmh(v_in)
        if math.isnan(v_target) or v_target <= 0:
            st.error("Невалидна скорост.")
        else:
            # намираме s, където v_personal(s) ~ v_target
            s_min, s_max = float(s_tab[0]), float(s_tab[-1])
            s_search = np.linspace(s_min, s_max, 1200)
            v_id_arr = lin_interp_extrap(s_search, s_tab, v_tab)
            r_arr = r_of_s(s_search)
            v_p_arr = np.maximum(v_id_arr * r_arr, 1e-9)
            idx = int(np.argmin(np.abs(v_p_arr - v_target)))
            s_sol = float(s_search[idx])
            v_p = float(v_p_arr[idx])
            t_p = (s_sol / v_p) * 60.0
            st.write(f"**Персонална дистанция:** {s_sol:.3f} km")
            st.write(f"**Персонално време:** {format_minutes_to_hms(t_p)}")
            st.caption(f"(Персонална скорост на тази дистанция: {v_p:.3f} km/h)")
# ===================== /ADDON =====================

