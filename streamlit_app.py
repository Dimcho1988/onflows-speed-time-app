import streamlit as st
import pandas as pd
import numpy as np
from model import IdealModel, PersonalModel
from cs import compute_cs_wprime, format_cs, format_wprime

st.set_page_config(page_title="Speed–Time Model • onFlows", layout="wide")

st.title("⚡ Индивидуален модел скорост–време (onFlows)")
st.caption("Линеен идеален модел + персонализация и отклонение (%). CS & W′ от 3′ и 12′.")

# ---------- Data loading ----------
st.sidebar.header("1) Идеални данни")

uploaded = st.sidebar.file_uploader("Качи CSV с колони distance_km, time_min", type=["csv"])
if uploaded:
    ideal_df = pd.read_csv(uploaded)
else:
    # Чете локалния файл, ако е в репото. Иначе ползва примерните данни.
    try:
        ideal_df = pd.read_csv("ideal_distance_time_speed.csv")
    except Exception:
        ideal_df = pd.DataFrame({
            "distance_km":[1, 3, 5, 10, 15, 21.097, 42.195],
            "time_min":[4, 12, 22, 46, 72, 110, 230]
        })

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

default_points = pd.DataFrame({"distance_km":[1.0, 3.0, 10.0], "v_real_kmh":[np.nan, np.nan, np.nan]})
user_points_df = st.sidebar.data_editor(default_points, num_rows="dynamic", key="user_points").dropna()
user_points = list(zip(user_points_df["distance_km"].tolist(), user_points_df["v_real_kmh"].tolist()))
personal = PersonalModel.from_user_points(ideal, user_points)

# ---------- Global deviation ----------
st.sidebar.header("3) Отклонение от идеала (%)")
deviation_pct = st.sidebar.number_input(
    "Положителна стойност = по-бавно (по-слабо) от идеала",
    value=0.0, step=0.5, format="%.1f"
)
# множител за скоростта (например 10% по-бавно => 0.90)
speed_factor = max(1e-6, 1.0 - deviation_pct/100.0)

# Помощни функции с отклонение
def v_with_dev(v_kmh: np.ndarray) -> np.ndarray:
    return v_kmh * speed_factor

def t_from_s_and_v(s_km: np.ndarray, v_kmh: np.ndarray) -> np.ndarray:
    v = np.maximum(v_kmh, 1e-9)
    return 60.0 * s_km / v

def time_with_dev_for_distance(s_km: float) -> float:
    """Време за s при персонален модел + отклонение."""
    v = personal.speed_for_distance(s_km) * speed_factor
    v = max(v, 1e-9)
    return 60.0 * s_km / v

def distance_for_time_with_dev(t_min: float, s_min: float = 0.01, s_max: float = 100.0) -> float:
    """Намира s така, че time_with_dev_for_distance(s) ≈ t_min (бисекция)."""
    lo, hi = s_min, s_max
    def f(s): return time_with_dev_for_distance(s) - t_min
    # разшири граници при нужда
    while f(hi) < 0 and hi < 1e6:
        hi *= 2
    while f(lo) > 0 and lo > 1e-6:
        lo *= 0.5
    for _ in range(120):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if abs(fm) < 1e-6:
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)

# ---------- CS & W′ ----------
st.sidebar.header("4) Critical Speed (CS) и W'")
st.sidebar.write("Скорости при 3 и 12 мин от персоналния модел (с отклонението). Можеш да ги променяш.")

def model_speed_at_time_with_dev(t_min: float) -> float:
    s = distance_for_time_with_dev(t_min, s_min=0.05, s_max=float(ideal_df["distance_km"].max())*2)
    return personal.speed_for_distance(s) * speed_factor

v3_default = model_speed_at_time_with_dev(3.0)
v12_default = model_speed_at_time_with_dev(12.0)

col_a, col_b = st.sidebar.columns(2)
v3 = col_a.number_input("v(3 мин) km/h", value=round(v3_default, 2))
v12 = col_b.number_input("v(12 мин) km/h", value=round(v12_default, 2))

try:
    CS_mps, W_m = compute_cs_wprime([(3.0, v3), (12.0, v12)])
    st.sidebar.success(f"CS ≈ {format_cs(CS_mps):.2f} km/h | W' ≈ {format_wprime(W_m):.0f} m")
except Exception as e:
    st.sidebar.error(f"CS/W' грешка: {e}")
    CS_mps, W_m = np.nan, np.nan

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Криви", "🔮 Прогнози", "🧮 Таблици"])

# Мрежа по дистанция
s_grid = np.linspace(ideal_df["distance_km"].min(), ideal_df["distance_km"].max(), 200)
v_id = personal.ideal.v_ideal(s_grid)
t_id = personal.ideal.t_ideal(s_grid)
v_p = personal.v_personal(s_grid)
t_p = personal.t_personal(s_grid)

# Прилага отклонението
v_p_dev = v_with_dev(v_p)
t_p_dev = t_from_s_and_v(s_grid, v_p_dev)

with tab1:
    st.subheader("Скорост спрямо дистанция")
    chart_v = pd.DataFrame({
        "s_km": s_grid,
        "v_ideal": v_id,
        "v_personal (без откл.)": v_p,
        f"v_personal (с {deviation_pct:.1f}% откл.)": v_p_dev
    }).set_index("s_km")
    st.line_chart(chart_v)
    st.subheader("Време спрямо дистанция")
    chart_t = pd.DataFrame({
        "s_km": s_grid,
        "t_ideal": t_id,
        "t_personal (без откл.)": t_p,
        f"t_personal (с {deviation_pct:.1f}% откл.)": t_p_dev
    }).set_index("s_km")
    st.line_chart(chart_t)

with tab2:
    st.subheader("Прогноза по въведени параметри (включва отклонението)")
    mode = st.radio("Избери вход:", ["Дистанция (km)", "Време (min)", "Скорост (km/h)"], horizontal=True)

    if mode == "Дистанция (km)":
        s_in = st.number_input("Дистанция (km)", value=5.0, min_value=0.05, step=0.05)
        v_out = personal.speed_for_distance(s_in) * speed_factor
        t_out = t_from_s_and_v(np.array([s_in]), np.array([v_out]))[0]
        st.info(f"Скорост: **{v_out:.2f} km/h**, Време: **{t_out:.2f} min**")

    elif mode == "Време (min)":
        t_in = st.number_input("Време (min)", value=30.0, min_value=0.1, step=0.1)
        s_out = distance_for_time_with_dev(t_in, s_min=0.05, s_max=float(ideal_df['distance_km'].max())*2)
        v_out = personal.speed_for_distance(s_out) * speed_factor
        st.info(f"Дистанция: **{s_out:.2f} km**, Средна скорост: **{v_out:.2f} km/h**")

    else:
        v_in = st.number_input("Скорост (km/h)", value=12.0, min_value=0.1, step=0.1)
        # търси s, при която персоналната скорост с отклонение е най-близка до v_in
        s_candidates = np.linspace(ideal_df["distance_km"].min(), ideal_df["distance_km"].max(), 600)
        v_candidates = v_with_dev(personal.v_personal(s_candidates))
        idx = int(np.argmin(np.abs(v_candidates - v_in)))
        s_out = float(s_candidates[idx])
        t_out = t_from_s_and_v(np.array([s_out]), np.array([v_candidates[idx]]))[0]
        st.info(f"Най-близка дистанция: **{s_out:.2f} km**, Очаквано време: **{t_out:.2f} min**")

with tab3:
    st.subheader("Идеал vs персонален модел (с/без отклонение)")
    table_df = pd.DataFrame({
        "distance_km": s_grid,
        "t_ideal_min": t_id,
        "v_ideal_kmh": v_id,
        "t_personal_min": t_p,
        "v_personal_kmh": v_p,
        f"v_personal_with_dev({deviation_pct:.1f}%)": v_p_dev,
        f"t_personal_with_dev({deviation_pct:.1f}%)": t_p_dev,
        "ratio_r (v_personal/v_ideal)": np.divide(v_p, v_id, out=np.ones_like(v_p), where=v_id>1e-12),
        f"ratio_with_dev({deviation_pct:.1f}%)": np.divide(v_p_dev, v_id, out=np.ones_like(v_p_dev), where=v_id>1e-12)
    })
    st.dataframe(table_df.round(3), use_container_width=True, hide_index=True)

st.divider()
st.caption("© onFlows • Отклонението променя скоростта (v · (1 − p/100)), а времето се преизчислява.")
