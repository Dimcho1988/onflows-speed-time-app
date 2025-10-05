import pandas as pd
from .base import DataProvider

class CSVProvider(DataProvider):
    def __init__(self, path="ideal_distance_time_speed.csv"):
        self.path = path
    def load_ideal(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df[["distance_km","time_min"]].dropna().sort_values("distance_km")
