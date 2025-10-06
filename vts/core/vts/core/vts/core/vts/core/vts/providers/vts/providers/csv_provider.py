import pandas as pd
from vts.providers.base import DataProvider

class CSVProvider(DataProvider):
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        return pd.read_csv(self.filepath)
