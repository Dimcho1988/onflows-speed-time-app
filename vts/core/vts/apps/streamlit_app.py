from vts.core.critical_speed import compute_critical_speed, apply_wprime_correction

# 1️⃣ Изчисляваме скорости от индивидуалната крива за 3 и 12 минути
v1 = personal_curve.get_speed_from_time(3)   # km/h
v2 = personal_curve.get_speed_from_time(12)  # km/h

# 2️⃣ Изчисляваме критична скорост и W'
CS, W_prime = compute_critical_speed(3, v1, 12, v2)

# 3️⃣ Сравняваме с идеала
CS_ideal, W_prime_ideal = compute_critical_speed(3, ideal_curve.get_speed_from_time(3),
                                                 12, ideal_curve.get_speed_from_time(12))
k = W_prime / W_prime_ideal

# 4️⃣ Коригираме индивидуалната функция r(s)
r_corrected = apply_wprime_correction(distances, r_values, k)

# 5️⃣ Показваме резултати в интерфейса
st.markdown("### ⚙️ Критична скорост и анаеробен резерв")
st.write(f"**Критична скорост (CS):** {CS*3.6:.2f} km/h")
st.write(f"**Анаеробен резерв (W′):** {W_prime/1000:.2f} kJ (отн.)")
st.write(f"**Съотношение към идеала (k):** {k:.2f}")

# 6️⃣ Актуализираме графиката с новата r'(s)
st.line_chart({
    "Оригинален процент r(s)": r_values,
    "Коригиран процент r'(s)": r_corrected
})
