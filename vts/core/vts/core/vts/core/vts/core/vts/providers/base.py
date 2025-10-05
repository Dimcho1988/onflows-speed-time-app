from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def load_ideal(self) -> pd.DataFrame: ...
