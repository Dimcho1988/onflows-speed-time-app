import streamlit as st, pandas as pd, numpy as np
from vts.core.parsing import parse_distance_km, parse_speed_kmh, parse_time_min
from vts.core.metrics import min_to_hms
from vts.core.ideal import IdealCurve
from vts.core.personalize import PersonalizedCurve
from vts.providers.csv_provider import CSVProvider

st.set_page_config(page_title="VTS Model", layout="wide")
st.title("VTS Model — Ideal vs Personalized")

# 1) данни
uploaded = st.file_uploader("Качи CSV (distance_km,time_min) или остави празно за вграден файл", type=["csv"])
if uploaded is not None: df = pd.read_csv(uploaded)
else: df = CSVProvider().load_ideal()

s_tab = df["distance_km"].values.astype(float)
t_tab = df["time_min"].values.astype(float)
ideal = IdealCurve(s_tab, t_tab)

# 2) опорни точки
st.sidebar.header("Опорни точки (distance + speed)")
n = st.sidebar.number_input("брой точки", 1, 10, 3)
anchors = []
for i in range(n):
    d = st.sidebar.text_input(f"[{i+1}] distance", value="" if i>=2 else ["2km","13km",""][i])
    v = st.sidebar.text_input(f"[{i+1}] speed",    value="" if i>=2 else ["16","13",""][i])
    if d and v:
        dk, vk = parse_distance_km(d), parse_speed_kmh(v)
        if not np.isnan(dk) and not np.isnan(vk):
            anchors.append((dk, vk))

curve = ideal if len(anchors)==0 else PersonalizedCurve.from_sv(ideal, anchors)

# 3) таблица 100m→маратон + % отклонение
dist = np.array([0.1,0.2,0.3,0.4,0.5,0.8,1.0,1.5,2.0,3.0,5.0,8.0,10.0,12.0,15.0,16.09,20.0,21.097,25.0,30.0,32.18,35.0,40.0,42.195])
rows=[]
for s in dist:
    id_v, id_t = ideal.from_distance(s)["speed_kmh"], ideal.from_distance(s)["time_min"]
    pr = curve.from_distance(s)
    pct = 100.0*pr["speed_kmh"]/id_v
    rows.append({
        "Дистанция (km)": s,
        "Идеална скорост (km/h)": id_v,
        "Персонална скорост (km/h)": pr["speed_kmh"],
        "Идеално време": min_to_hms(id_t),
        "Персонално време": min_to_hms(pr["time_min"]),
        "% от идеала (скорост)": pct
    })
df_out = pd.DataFrame(rows)
st.dataframe(df_out, use_container_width=True)

st.subheader("Отклонение от идеала (скорост, %)")
st.line_chart(df_out.set_index("Дистанция (km)")["% от идеала (скорост)"])

# 4) Бързи заявки
st.subheader("Бързи заявки")
c1,c2,c3 = st.columns(3)
with c1:
    d_in = st.text_input("distance", "1km")
    if st.button("От дистанция"):
        s = parse_distance_km(d_in); res = curve.from_distance(s)
        st.write({"speed_kmh": round(res["speed_kmh"],3), "time": min_to_hms(res["time_min"]), "distance_km": s})
with c2:
    t_in = st.text_input("time (min or mm:ss)", "30")
    if st.button("От време"):
        tm = parse_time_min(t_in); res = curve.from_time(tm)
        st.write({"distance_km": round(res["distance_km"],3), "speed_kmh": round(res["speed_kmh"],3)})
with c3:
    v_in = st.text_input("speed (km/h or 5:00 min/km)", "22")
    if st.button("От скорост"):
        vk = parse_speed_kmh(v_in); res = curve.from_speed(vk)
        st.write({"distance_km": round(res["distance_km"],3), "time": min_to_hms(res["time_min"])})
