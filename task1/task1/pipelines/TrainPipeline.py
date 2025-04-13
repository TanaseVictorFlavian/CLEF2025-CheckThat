from task1.config import ProjectPaths
import pandas as pd
from task1.pipelines.IngestionPipeline import IngestionPipeline
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Any
from abc import ABC, abstractmethod


class TrainPipeline():
    
    def __init__(
        self, 
        language: str, 
        model: Any, 
        data_path = None, 
        data: IngestionPipeline = None,
        batch_size: int = 128,
    ):
        self.language = language
        self.model = model
        self.data_path = data_path
        self.data = data
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
    
    def unpack_data(self):
        # Unpack data from parquet files
        self.train_data = pd.read_parquet(self.data_path / f"train_{self.language}_embeddings.parquet") 
        self.val_data = pd.read_parquet(self.data_path / f"dev_{self.language}_embeddings.parquet")
        
    def split_data(self):
        if self.data is None:
            self.unpack_data()
            
        # Convert embeddings from lists to numpy arrays
        self.X_train = np.array(self.train_data["embedding"].tolist())
        self.y_train = self.train_data["label"].values
        self.X_val = np.array(self.val_data["embedding"].tolist())
        self.y_val = self.val_data["label"].values
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

        

        
