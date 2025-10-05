import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

# === Настройки на страницата ===
st.set_page_config(page_title="VTS Predictor — Ideal vs Personalized", layout="wide")

# === Помощни функции ===
def parse_distance_to_km(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().lower().replace(" ", "")
    if not s:
        return math.nan
    if s in {"marathon", "maraton"}:
        return 42.195
    if s in {"half", "halfmarathon", "polumaraton"}:
        return 21.097
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
    if "km/h" in s:
        return float(s.replace("km/h", ""))
    if "kph" in s:
        return float(s.replace("kph", ""))
    if ":" in s:  # мин/км
        parts = s.split(":")
        if len(parts) == 2:
            mins, secs = parts
            pace_min = float(mins) + float(secs) / 60.0
            return 60.0 / pace_min
    if s.replace(".", "", 1).isdigit():
        return float(s)
    return math.nan


def format_minutes_to_hms(minutes: float) -> str:
    """Преобразува време от минути във формат Ч:ММ:СС."""
    if pd.isna(minutes) or minutes <= 0:
        return "-"
    total_seconds = int(minutes * 60)
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d} ч"
    else:
        return f"{mins}:{secs:02d} мин"


# === Зареждане на идеалната крива ===
st.title("🏃‍♂️ VTS Predictor — Ideal vs Personalized")

st.markdown("""
1. Зареди **идеалната крива** (CSV с колони `distance_km`, `time_min`).
2. Въведи **реални точки** на твоя атлет.
3. Ще получиш прогнозен профил и процент от идеала.
""")

uploaded_file = st.file_uploader("Качи CSV файл или остави празно (използва се вградената крива):", type=["csv"])

if uploaded_file is not None:
    df_ideal = pd.read_csv(uploaded_file)
else:
    df_ideal = pd.read_csv("ideal_distance_time_speed.csv")

# изчисляваме идеална скорост
df_ideal["speed_kmh"] = df_ideal["distance_km"] / (df_ideal["time_min"] / 60)

# създаваме интерполации
ideal_speed_from_dist = interp1d(df_ideal["distance_km"], df_ideal["speed_kmh"], fill_value="extrapolate")
ideal_time_from_dist = interp1d(df_ideal["distance_km"], df_ideal["time_min"], fill_value="extrapolate")

# === Въвеждане на персонални данни ===
st.sidebar.header("Персонализация (опорни точки)")

point_type = st.sidebar.radio("Тип точки", ["distance + speed", "time + speed", "time + distance"])
num_points = st.sidebar.number_input("Брой точки", 1, 5, 3)

points = []
for i in range(num_points):
    st.sidebar.markdown(f"**[{i+1}] точка**")
    if point_type == "distance + speed":
        d = st.sidebar.text_input(f"[{i+1}] distance (km, k, m)", "")
        v = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", "")
        if d and v:
            dk = parse_distance_to_km(d)
            vk = parse_speed_to_kmh(v)
            points.append((dk, vk))
    elif point_type == "time + speed":
        t = st.sidebar.number_input(f"[{i+1}] time (мин)", 0.0)
        v = st.sidebar.text_input(f"[{i+1}] speed (km/h или 5:00 min/km)", "")
        if t and v:
            vk = parse_speed_to_kmh(v)
            dk = np.interp(t, df_ideal["time_min"], df_ideal["distance_km"])
            points.append((dk, vk))
    else:
        t = st.sidebar.number_input(f"[{i+1}] time (мин)", 0.0)
        d = st.sidebar.text_input(f"[{i+1}] distance (km, k, m)", "")
        if t and d:
            dk = parse_distance_to_km(d)
            vk = dk / (t / 60)
            points.append((dk, vk))

if len(points) >= 2:
    points = sorted(points, key=lambda x: x[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    personal_speed = interp1d(x, y, fill_value="extrapolate")
    personal_factor = np.array([personal_speed(d) / ideal_speed_from_dist(d) for d in x])
    factor_interp = interp1d(x, personal_factor, fill_value=(personal_factor[0], personal_factor[-1]))

    distances = np.linspace(min(df_ideal["distance_km"]), max(df_ideal["distance_km"]), 40)
    data = []
    for d in distances:
        ideal_v = float(ideal_speed_from_dist(d))
        personal_v = ideal_v * float(factor_interp(d))
        personal_t = (d / personal_v) * 60
        data.append({
            "distance_km": d,
            "ideal_speed_kmh": ideal_v,
            "personal_speed_kmh": personal_v,
            "personal_time_min": personal_t,
            "percent_of_ideal": personal_v / ideal_v * 100
        })

    df_results = pd.DataFrame(data)
    df_results["време (ч:мм:сс)"] = df_results["personal_time_min"].apply(format_minutes_to_hms)

    st.subheader("📊 Резултати:")
    st.dataframe(df_results[[
        "distance_km", "ideal_speed_kmh", "personal_speed_kmh", "време (ч:мм:сс)", "percent_of_ideal"
    ]].rename(columns={
        "distance_km": "Дистанция (km)",
        "ideal_speed_kmh": "Идеална скорост (km/h)",
        "personal_speed_kmh": "Персонална скорост (km/h)",
        "percent_of_ideal": "% от идеала"
    }), use_container_width=True)

    # бутон за сваляне
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Изтегли резултатите (CSV)", csv, "vts_results.csv", "text/csv")
else:
    st.info("Въведи поне две опорни точки, за да се изчисли персонализираният модел.")

