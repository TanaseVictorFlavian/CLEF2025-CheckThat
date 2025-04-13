from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class EvaluationPipeline(ABC):
    def __init__(
        self, model, paths: Path, data=None, language: str = "en", batch_size: int = 128
    ):
        self.model = model
        self.data_path = paths
        self.data = data
        self.language = language
        self.batch_size = batch_size
        self.test_loader = None

    def unpack_data(self):
        # Unpack data from parquet files
        self.test_data = pd.read_parquet(self.data_path / f"dev_test_{self.language}_embeddings.parquet")

    def split_data(self):
        if self.data is None:
            self.unpack_data()
            
        self.X_test = np.array(self.test_data["embedding"].tolist())
        self.y_test = np.array(self.test_data["label"].values)

    @abstractmethod
    def evaluate_model(self):
        pass

