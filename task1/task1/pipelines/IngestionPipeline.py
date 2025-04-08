from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np

class IngestionPipeline:
    
    def __init__(self, data_path : Path, language : str, embedding_model):
        self.data_path = data_path
        self.language = language
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.embedding_model = embedding_model
        self.load_data()
        
    
    def run(self):
        self.train_data = self.preprocess_data(self.train_data)
        self.val_data = self.preprocess_data(self.val_data)
        self.test_data = self.preprocess_data(self.test_data)
        
        # Save train data
        pd.DataFrame({
            "embedding": self.train_data[0].tolist(),  # Convert numpy array to list
            "label": self.train_data[1]
        }).to_parquet(self.data_path / f"train_{self.language}_embeddings.parquet", index=False)
        
        # Save validation data
        pd.DataFrame({
            "embedding": self.val_data[0].tolist(),
            "label": self.val_data[1]
        }).to_parquet(self.data_path / f"dev_{self.language}_embeddings.parquet", index=False)
        
        # Save test data
        pd.DataFrame({
            "embedding": self.test_data[0].tolist(),
            "label": self.test_data[1]
        }).to_parquet(self.data_path / f"dev_test_{self.language}_embeddings.parquet", index=False)
    
    def load_data(self):
        self.train_data = pd.read_csv(self.data_path / f"train_{self.language}.tsv", sep='\t')
        self.val_data = pd.read_csv(self.data_path / f"dev_{self.language}.tsv", sep='\t')
        self.test_data = pd.read_csv(self.data_path / f"dev_test_{self.language}.tsv", sep='\t')
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data by getting embeddings and mapping labels to binary values.
        
        Args:
            data: Input DataFrame containing 'sentence' and 'label' columns
            
        Returns:
            tuple: (embeddings, binary_labels)
        """
        # Get embeddings from sentences
        embeddings = self.embedding_model.encode(data['sentence'].values)
        
        # Map labels to binary values (OBJ -> 0, SUBJ -> 1)
        binary_labels = np.where(data['label'].values == 'SUBJ', 1, 0)
        
        return embeddings, binary_labels
    




