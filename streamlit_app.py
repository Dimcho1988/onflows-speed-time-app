import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.model import IdealModel, PersonalModel
from src.cs import compute_cs_wprime, format_cs, format_wprime

st.set_page_config(page_title="Speed–Time Model • onFlows", layout="wide")

st.title("⚡ Индивидуален модел скорост–време (onFlows)")
st.caption("Лек модел с интерполация + персонализация и изчисляване на Critical Speed (CS) и W'.")

# ---------- Data loading ----------
st.sidebar.header("1) Идеални данни")
default_csv_path = Path("data/ideal_distance_time_speed.csv")

# ✅ поправен ред – има затваряща скоба
uploaded = st.sidebar.file_uploader("Качи CSV с колони distance_km, time_min", type=["csv"])
if uploaded:
    ideal_df = pd.read_csv(uploaded)
else:
    if default_csv_path.exists():
        ideal_df = pd.read_csv(default_csv_path)
    else:
        st.error("Липсва CSV с идеални данни. Качи файл в sidebar-а.")
        st.stop()

# Validate basic schema
required_cols = {"distance_km", "time_min"}
if not required_cols.issubset(set(ideal_df.columns)):
    st.error(f"CSV трябва да съдържа колоните: {required_cols}. Открити: {list(ideal_df.columns)}")
    st.stop()

ideal_df = ideal_df.dropna().sort_values("distance_km")
ideal_df["v_kmh"] = ideal_df["distance_km"] / (ideal_df["time_min"]/60.0)

st.dataframe(ideal_df, use_container_width=True, hide_index=True)

ideal = IdealModel.from_distance_time_points(list(zip(ideal_df["distance_km"], ideal_df["time_min"])))

# ---------- Personalization points ----------
st.sidebar.header("2) Персонализация (по избор)")
st.sidebar.write("Добави реални тестове като двойки `(distance_km, v_real_kmh)`.")

default_points = pd.DataFrame({
    "distance_km":[1.0, 3.0, 10.0],
    "v_real_kmh":[np.nan, np.nan, np.nan]
})
user_points_df = st.sidebar.data_editor(default_points, num_rows="dynamic", key="user_points")
user_points_df = user_points_df.dropna()

user_points = list(zip(user_points_df["distance_km"].tolist(), user_points_df["v_real_kmh"].tolist()))
personal = PersonalModel.from_user_points(ideal, user_points)

# ---------- CS & W' from 3' and 12' ----------
st.sidebar.header("3) Critical Speed (CS) и W'")
st.sidebar.write("По подразбиране скоростите за 3 и 12 мин се оценяват от персоналния модел. Можеш да ги презапишеш.")

def model_speed_at_time(model: PersonalModel, t_min: float) -> float:
    s = model.distance_for_time(t_min, s_min=0.05, s_max=float(ideal_df['distance_km'].max())*2)
    v = model.speed_for_distance(s)
    return float(v)

v3_default = model_speed_at_time(personal, 3.0)
v12_default = model_speed_at_time(personal, 12.0)

col_a, col_b = st.sidebar.columns(2)
v3 = col_a.number_input("v(3 мин) km/h", value=round(v3_default, 2))
v12 = col_b.number_input("v(12 мин) km/h", value=round(v12_default, 2))

try:
    CS_mps, W_m = compute_cs_wprime([(3.0, v3), (12.0, v12)])
    st.sidebar.success(f"CS ≈ {format_cs(CS_mps):.2f} km/h | W' ≈ {format_wprime(W_m):.0f} m")
except Exception as e:
    st.sidebar.error(f"CS/W' грешка: {e}")
    CS_mps, W_m = np.nan, np.nan

# ---------- Main tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Криви", "🔮 Прогнози", "🧮 Таблици"])

# Shared s-grid
s_grid = np.linspace(ideal_df["distance_km"].min(), ideal_df["distance_km"].max(), 200)
v_id = personal.ideal.v_ideal(s_grid)
t_id = personal.ideal.t_ideal(s_grid)
v_p = personal.v_personal(s_grid)
t_p = personal.t_personal(s_grid)

with tab1:
    st.subheader("Скорост спрямо дистанция")
    st.line_chart(pd.DataFrame({"s_km": s_grid, "v_ideal": v_id, "v_personal": v_p}).set_index("s_km"))
    st.subheader("Време спрямо дистанция")
    st.line_chart(pd.DataFrame({"s_km": s_grid, "t_ideal": t_id, "t_personal": t_p}).set_index("s_km"))

with tab2:
    st.subheader("Прогноза по въведени параметри")
    mode = st.radio("Избери вход:", ["Дистанция (km)", "Време (min)", "Скорост (km/h)"], horizontal=True)

    if mode == "Дистанция (km)":
        s_in = st.number_input("Дистанция (km)", value=5.0, min_value=0.05, step=0.05)
        v_out = personal.speed_for_distance(s_in)
        t_out = personal.time_for_distance(s_in)
        st.info(f"Скорост: **{v_out:.2f} km/h**, Време: **{t_out:.2f} min**")

    elif mode == "Време (min)":
        t_in = st.number_input("Време (min)", value=30.0, min_value=0.1, step=0.1)
        s_out = personal.distance_for_time(t_in, s_min=0.05, s_max=float(ideal_df['distance_km'].max())*2)
        v_out = personal.speed_for_distance(s_out)
        st.info(f"Дистанция: **{s_out:.2f} km**, Средна скорост: **{v_out:.2f} km/h**")

    else:
        v_in = st.number_input("Скорост (km/h)", value=12.0, min_value=0.1, step=0.1)
        s_candidates = np.linspace(ideal_df["distance_km"].min(), ideal_df["distance_km"].max(), 500)
        v_candidates = personal.v_personal(s_candidates)
        idx = int(np.argmin(np.abs(v_candidates - v_in)))
        s_out = float(s_candidates[idx])
        t_out = personal.time_for_distance(s_out)
        st.info(f"Най-близка дистанция: **{s_out:.2f} km**, Очаквано време: **{t_out:.2f} min**")

with tab3:
    st.subheader("Идеал срещу персонален модел (по мрежа от дистанции)")
    table_df = pd.DataFrame({
        "distance_km": s_grid,
        "t_ideal_min": t_id,
        "v_ideal_kmh": v_id,
        "t_personal_min": t_p,
        "v_personal_kmh": v_p,
        "ratio_r": np.divide(v_p, v_id, out=np.ones_like(v_p), where=v_id>1e-12)
    })
    st.dataframe(table_df.round(3), use_container_width=True, hide_index=True)

st.divider()
st.caption("© onFlows • Модел: линейна интерполация върху идеалните точки + r(s) от реални тестове. CS/W' от 3' и 12'.")
