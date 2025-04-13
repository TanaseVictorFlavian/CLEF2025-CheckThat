from task1.pipelines.EvaluationPipeline import EvaluationPipeline
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F


class EvaluationPipelineNN(EvaluationPipeline):
    def __init__(self, model, paths: Path, data):
        super().__init__(model, paths, data)

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
    
    def evaluate_model(self, criterion):
        self.model.eval()
        total_loss = 0
        y_pred = []
    
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                y_pred.append(self.get_predictions(logits))
       
        self.test_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {self.test_loss}")
        self.compute_metrics(np.concatenate(y_pred))
        
    def get_stats(self):
        stats = {
            "test_loss": self.test_loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }
        return stats
