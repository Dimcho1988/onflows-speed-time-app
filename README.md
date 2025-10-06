
# onFlows • Индивидуален модел скорост–време

Леко Streamlit приложение за:
- Изграждане на идеална крива `t(s)` и `v(s)` от CSV.
- Персонализация чрез реални тестови точки `(distance_km, v_real_kmh)`.
- Прогнози по дадена дистанция, време или скорост.
- Изчисляване на Critical Speed (CS) и W' по 3' и 12' (по Monod).

## Структура

```
onflows_speed_time_app/
├─ streamlit_app.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ ideal_distance_time_speed.csv
└─ src/
   ├─ __init__.py
   ├─ model.py
   └─ cs.py
```

## Разгръщане (GitHub → Streamlit Cloud)

1. Създай нов GitHub репо и качи съдържанието на папката.
2. В Streamlit Community Cloud: **New app** → избери репото, главния файл `streamlit_app.py`.
3. Готово. По желание замени `data/ideal_distance_time_speed.csv` с твоите идеални точки.

## CSV формат

`data/ideal_distance_time_speed.csv` трябва да съдържа поне две колони:

- `distance_km` – дистанция в километри (възходящо)
- `time_min` – време в минути

Скоростта се смята автоматично като `distance_km / (time_min/60)`.

## CS & W'

В Sidebar приложението извлича скоростите при 3 и 12 минути от персоналния модел (или въвеждаш ръчно)
и изчислява CS (в km/h) и W' (в метри) по линейния модел `d = CS*t + W'`.
