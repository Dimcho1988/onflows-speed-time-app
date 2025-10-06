import pandas as pd

def load_ideal_curve(filepath="ideal_distance_time_speed.csv"):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    if "distance_km" not in df or "time_min" not in df:
        raise ValueError("CSV трябва да съдържа колони distance_km и time_min.")
    df["speed_kmh"] = df["distance_km"] / (df["time_min"] / 60.0)
    return df
