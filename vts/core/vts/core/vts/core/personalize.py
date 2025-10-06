import pandas as pd
import numpy as np

def build_personal_curve(points):
    df = pd.DataFrame(points, columns=["distance_km", "speed_kmh"])
    df = df.dropna().sort_values("distance_km")

    # интерполация на време
    df["time_min"] = df["distance_km"] / df["speed_kmh"] * 60.0
    return df

def deviation_from_ideal(ideal_df, personal_df):
    merged = pd.merge(ideal_df, personal_df, on="distance_km", suffixes=("_ideal", "_personal"), how="inner")
    merged["r_speed"] = merged["speed_kmh_personal"] / merged["speed_kmh_ideal"] * 100
    return merged
