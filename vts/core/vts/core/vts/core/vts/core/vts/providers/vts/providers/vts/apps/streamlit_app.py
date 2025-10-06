import streamlit as st
import pandas as pd
import numpy as np
from vts.core.ideal import load_ideal_curve
from vts.core.personalize import build_personal_curve, deviation_from_ideal
from vts.core.critical_speed import compute_critical_speed, apply_wprime_correction

# Настройки
st.set_page_config(page_title="VTS Predictor — Ideal vs Personalized", layout="wide")
st.title("🏃‍♂️ VTS Predictor — Ideal vs Personalized")

# Зареждане на идеалната крива
ideal_df = load_ideal_curve("ideal_distance_time_speed.csv")

# Въвеждане на опорни точки
st.sidebar.header("Персонализация")
n = st.sidebar.number_input("Брой точки", 2, 10, 3)
points = []
for i in range(n):
    d = st.sidebar.text_input(f"[{i+1}] дистанция (km, m)", value=f"{i+1}km")
    v = st.sidebar.text_input(f"[{i+1}] скорост (km/h)", value="15")
    try:
        d = float(d.replace("km",""))
        v = float(v)
        points.append((d, v))
    except:
        pass

if points:
    personal_df = build_personal_curve(points)
    dev_df = deviation_from_ideal(ideal_df, personal_df)
    st.subheader("📊 Сравнение с идеала")
    st.dataframe(dev_df[["distance_km", "speed_kmh_ideal", "speed_kmh_personal", "r_speed"]])

    # CS и W'
    try:
        v1 = float(personal_df.iloc[(personal_df["time_min"]-3).abs().idxmin()]["speed_kmh"])
        v2 = float(personal_df.iloc[(personal_df["time_min"]-12).abs().idxmin()]["speed_kmh"])
        CS, Wp = compute_critical_speed(3, v1, 12, v2)

        v1i = float(ideal_df.iloc[(ideal_df["time_min"]-3).abs().idxmin()]["speed_kmh"])
        v2i = float(ideal_df.iloc[(ideal_df["time_min"]-12).abs().idxmin()]["speed_kmh"])
        CS_i, Wp_i = compute_critical_speed(3, v1i, 12, v2i)
        k = Wp / Wp_i if Wp_i != 0 else 1

        st.subheader("⚙️ Критична скорост и анаеробен резерв")
        st.write(f"Критична скорост (CS): **{CS*3.6:.2f} km/h**")
        st.write(f"Анаеробен резерв (W′): **{Wp/1000:.2f} kJ/kg**")
        st.write(f"Съотношение към идеала (k): **{k:.2f}**")

        r_corrected = apply_wprime_correction(dev_df["distance_km"], dev_df["r_speed"], k)
        dev_df["r_corrected"] = r_corrected

        st.line_chart(dev_df.set_index("distance_km")[["r_speed", "r_corrected"]])
    except Exception as e:
        st.error(f"Грешка при изчисление: {e}")
