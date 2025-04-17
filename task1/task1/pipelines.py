from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Any
import torch
import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from task1.config import ProjectPaths


class IngestionPipeline:
    def __init__(self, data_path : Path, language : str, embedding_model):
        self.data_path = data_path
        self.language = language
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.embedding_model = embedding_model
        self.load_data()
        
    def run(self, save_embeddings : bool = True):
        self.train_data = self.preprocess_data(self.train_data)
        self.val_data = self.preprocess_data(self.val_data)
        self.test_data = self.preprocess_data(self.test_data)
        
        # Save train data
        if not save_embeddings:
            return
        
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
    

class TrainPipeline:
    def __init__(
        self,
        language: str,
        model: Any,
        data_path=None,
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
        self.train_data = None
        self.val_data = None
        self.split_data()
        self.random_seed = 42
        self.set_random_seeds()

    def unpack_data(self):
        # Unpack data from parquet files
        
        print(f"Loading training data from : {self.data_path / f'train_{self.language}_embeddings.parquet'}")
        
        self.train_data = pd.read_parquet(
            self.data_path / f"train_{self.language}_embeddings.parquet"
        )
        
        print(f"Loading validation data from : {self.data_path / f'dev_{self.language}_embeddings.parquet'}")
        self.val_data = pd.read_parquet(
            self.data_path / f"dev_{self.language}_embeddings.parquet"
        )

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
    
    @abstractmethod
    def set_random_seeds(self):
        pass

class TrainPipelineNN(TrainPipeline):
    def __init__(self, language, model, data_path, data=None, batch_size=128, model_hyperparams=None):
        super().__init__(language, model, data_path, data, batch_size)
        if model_hyperparams is None:
            self.hyperparams = {}
        else:
            self.hyperparams = model_hyperparams
        self.train_losses = []
        self.val_losses = []
        
    def set_random_seeds(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        

    def create_data_loaders(self):
        """Create PyTorch DataLoaders for training and validation."""
        self.X_train = torch.Tensor(self.X_train)
        self.y_train = torch.Tensor(self.y_train)
        self.X_val = torch.Tensor(self.X_val)
        self.y_val = torch.Tensor(self.y_val)
        
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()
    
    def validate(self, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.unsqueeze(1).to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        
        if self.hyperparams:
            epochs = self.hyperparams["epochs"]
            lr = self.hyperparams["lr"]
            weight_decay = self.hyperparams["weight_decay"]
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            epochs = 5
            lr = 3e-4
            loss_fn = torch.nn.BCEWithLogitsLoss()
            weight_decay = 0
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

            self.hyperparams["epochs"] = epochs
            self.hyperparams["lr"] = lr
            self.hyperparams["weight_decay"] = weight_decay
            self.hyperparams["optimizer"] = "Adam"

        print(f"Training running on: {self.device}")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self.validate(loss_fn)
            
            # Store losses
            self.train_losses.append(train_loss / len(self.train_loader))
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Training Loss: {self.train_losses[-1]:.4f}")
            print(f"Validation Loss: {self.val_losses[-1]:.4f}")
            
            # Plot losses
            self.plot_losses()
        
        print("\nTraining completed!")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")
    
    def run(self):
        self.create_data_loaders()
        self.train()


class MasterPipeline:
    def __init__(self, config,  paths : ProjectPaths, ingestion_pipeline, training_pipeline, evaluation_pipeline):
        self.paths = paths
        self.ingestion_pipeline = ingestion_pipeline
        self.training_pipeline = training_pipeline
        self.evaluation_pipeline = evaluation_pipeline
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    
    def run(self, save_run_info: bool = True):
        self.ingestion_pipeline.run()
        self.training_pipeline.run()
        self.evaluation_pipeline.run()
        
        run_info =self.log_run()
        
        if save_run_info:
            with open(self.paths.run_info_path / f"run_{self.timestamp}.json", "w") as f:
                json.dump(run_info, f)
            
    def log_run(self):
        run_info = {
            "language": self.ingestion_pipeline.language,
            "hyperparams": self.training_pipeline.hyperparams,
            "model_arch": self.training_pipeline.model.get_architecture(),
            "stats": self.evaluation_pipeline.get_stats()
        }

        return run_info
        
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
        self.test_data = pd.read_parquet(
            self.data_path / f"dev_test_{self.language}_embeddings.parquet"
        )

    def split_data(self):
        if self.data is None:
            self.unpack_data()
            
        self.X_test = np.array(self.test_data["embedding"].tolist())
        self.y_test = np.array(self.test_data["label"].values)

    @abstractmethod
    def evaluate_model(self):
        pass
    
class EvaluationPipelineNN(EvaluationPipeline):
    def __init__(self, model, data_path, data):
        super().__init__(model, data_path, data)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def create_data_loaders(self):
        self.split_data()
        X_test = torch.Tensor(self.X_test)
        y_test = torch.Tensor(self.y_test)
        
        test_dataset = TensorDataset(X_test, y_test)
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

    def compute_metrics(self, y_pred):
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)
        
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"F1: {self.f1}")
    
    def get_predictions(self, logits):
        return [1 if logit > 0.5 else 0 for logit in F.sigmoid(logits)]
    
    def evaluate_model(self):
        self.model.eval()
        y_pred = []
    
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.unsqueeze(1).to(self.device)
                logits = self.model(batch_X)
                y_pred.append(self.get_predictions(logits))
       
        self.compute_metrics(np.concatenate(y_pred))
        
    def get_stats(self):
        stats = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }
        return stats

    def run(self):
        print("Running evaluation pipeline...")
        self.create_data_loaders()
        self.evaluate_model()
    