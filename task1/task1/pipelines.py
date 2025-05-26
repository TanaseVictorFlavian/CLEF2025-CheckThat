from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Any
import torch
import random
import os
from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sentence_transformers import SentenceTransformer
from task1.config import ProjectPaths

from task1.models.encoders import Encoder


torch.use_deterministic_algorithms(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

paths = ProjectPaths()

class IngestionPipeline:
    def __init__(self, language: list[str], encoder: Encoder):
        self.language = language
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.encoder = encoder
        self.load_data()

    def run(self, save_embeddings: bool = True):
        print("Running embedding model ...")
        self.train_data = self.encode_data(self.train_data)
        self.val_data = self.encode_data(self.val_data)
        self.test_data = self.encode_data(self.test_data)

        # Save train data
        if not save_embeddings:
            return
        
        # Create directory if it doesn't exist
        output_dir = paths.embeddings_dir / self.language / self.encoder.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(self.train_data).to_parquet(
            output_dir / "train.parquet", index=False
        )

        pd.DataFrame(self.val_data).to_parquet(
            output_dir / "dev.parquet", index=False
        )

        pd.DataFrame(self.test_data).to_parquet(
            output_dir / "test.parquet", index=False
        )

    def load_data(self):
        splits = os.listdir(paths.data_dir / self.language)
        train, val, test = splits[3], splits[0], splits[1]

        print(f"Loading training data from : {paths.data_dir / self.language / train}")
        self.train_data = pd.read_csv(
            paths.data_dir / self.language / train, sep="\t"
        )

        print(f"Loading validation data from : {paths.data_dir / self.language / val}")
        self.val_data = pd.read_csv(
            paths.data_dir / self.language / val, sep="\t"
        )

        print(f"Loading test data from : {paths.data_dir / self.language / test}")
        self.test_data = pd.read_csv(
            paths.data_dir / self.language / test, sep="\t"
        )

    def encode_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data by getting embeddings and mapping labels to binary values.

        Args:
            data: Input DataFrame containing 'sentence' and 'label' columns

        Returns:
            tuple: (embeddings, binary_labels)
        """
        # Get embeddings from sentences
        embeddings = self.encoder.encode(data["sentence"].values)

        # Map labels to binary values (OBJ -> 0, SUBJ -> 1)
        binary_labels = np.where(data["label"].values == "SUBJ", 1, 0)

        return {"embeddings": embeddings.tolist(), "labels": binary_labels.tolist()}


class TrainPipeline:
    def __init__(
        self,
        language: str,
        model: Any,
        data: Any = (None, None),
        batch_size: int = 128,
        use_class_weights: bool = True,
        random_seed: int = 42,
        encoder: Encoder = None,
    ):
        self.language = language
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.train_data, self.val_data = data
        self.random_seed = random_seed
        self.split_data()
        self.set_random_seeds()
        self.encoder = encoder
        if use_class_weights:
            self.set_class_weights()
        else:
            self.class_weights = None

    def unpack_data(self):
        if self.data is None:
            # Unpack data from parquet files
            print(
                f"Loading training data from : {paths.embeddings_dir / self.language / self.encoder.model_name / 'train.parquet'}"
            )

            self.train_data = pd.read_parquet(
                paths.embeddings_dir / self.language / self.encoder.model_name / "train.parquet"
            )

            print(
                f"Loading validation data from : {paths.embeddings_dir / self.language / self.encoder.model_name / 'dev.parquet'}"
            )
            self.val_data = pd.read_parquet(
                paths.embeddings_dir / self.language / self.encoder.model_name / "dev.parquet"
            )

        else:
            self.train_data = self.data[0]
            self.val_data = self.data[1]

    def split_data(self):
        self.unpack_data()

        # Convert embeddings from lists to numpy arrays
        self.X_train = np.array(self.train_data["embeddings"])
        self.y_train = np.array(self.train_data["labels"])
        self.X_val = np.array(self.val_data["embeddings"])
        self.y_val = np.array(self.val_data["labels"])

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def set_random_seeds(self):
        pass

    @abstractmethod
    def set_class_weights(self):
        pass

class TrainPipelineSklearn(TrainPipeline):
    def __init__(
        self,
        language,
        model,
        data=None,
        model_hyperparams=None,
        random_seed: int = 42,
        encoder: Encoder = None,
    ):
        super().__init__(
            language, model, data, use_class_weights=True, random_seed=random_seed
        )
        if model_hyperparams is None:
            self.hyperparams = {}
        else:
            self.hyperparams = model_hyperparams
        self.train_scores = []
        self.val_scores = []
        self.encoder = encoder

    def set_random_seeds(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
    def set_class_weights(self):
        classes = np.unique(self.y_train)
        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=self.y_train
        )
        self.class_weight_dict = dict(zip(classes, self.class_weights))

        # Apply class weights to model if supported
        if hasattr(self.model, "class_weight"):
            self.model.set_params(class_weight=self.class_weight_dict)
    
    def plot_scores(self):
        self.loss_figure = plt.figure(figsize=(10, 6))
        plt.plot(self.train_scores, label="Training Accuracy")
        plt.plot(self.val_scores, label="Validation Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.close()

    def train(self):
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        train_acc = accuracy_score(self.y_train, y_train_pred)
        val_acc = accuracy_score(self.y_val, y_val_pred)

        self.train_scores.append(train_acc)
        self.val_scores.append(val_acc)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        self.plot_scores()
        print("Training completed!")

    def run(self):
        self.train()
    
class TrainPipelineNN(TrainPipeline):
    def __init__(
        self,
        language,
        model,
        data=None,
        batch_size=128,
        model_hyperparams=None,
        random_seed: int = 42,
        encoder: Encoder = None,
    ):
        super().__init__(
            language, model, data, batch_size, random_seed=random_seed
        )
        if model_hyperparams is None:
            self.hyperparams = {}
        else:
            self.hyperparams = model_hyperparams
        self.train_losses = []
        self.val_losses = []
        self.class_weights = None
        self.encoder = encoder

    def set_random_seeds(self):
        """Set random seeds for all relevant libraries."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        # Also set cuda seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_class_weights(self):
        classes = np.unique(self.y_train)
        self.class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=self.y_train
        )

    def create_data_loaders(self):
        """Create PyTorch DataLoaders for training and validation."""
        self.X_train = torch.Tensor(self.X_train)
        self.y_train = torch.Tensor(self.y_train)
        self.X_val = torch.Tensor(self.X_val)
        self.y_val = torch.Tensor(self.y_val)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)

        # Create generator for reproducibility
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            generator=g,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            generator=g,
        )

    def plot_losses(self):
        self.loss_figure = plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
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
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        else:
            epochs = 10
            lr = 3e-4
            loss_fn = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
            weight_decay = 0
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

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
    def __init__(
        self,
        encoder: Encoder,
        classifier: Any,
        language: str,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ingestion_pipeline = IngestionPipeline(language=language, encoder=encoder)
        self.language = language
        self.classifier = classifier
        self.encoder = encoder

    def run(self, save_run_info: bool = True):
        self.ingestion_pipeline.run()
        self.training_pipeline = TrainPipelineNN(
            encoder=self.encoder,
            language=self.language,
            model=self.classifier,
            data=(self.ingestion_pipeline.train_data, self.ingestion_pipeline.val_data),
        )
        self.training_pipeline.run()
        self.evaluation_pipeline = EvaluationPipelineNN(
            model=self.training_pipeline.model,
            data=self.ingestion_pipeline.test_data,
            encoder =self.encoder,
            language=self.language,
        )
        self.evaluation_pipeline.run()
        self.log_run()

    def log_run(self, save_run_info: bool = True):
        run_info = {
            "language": self.language,
            "encoder": self.encoder.get_params(),
            "classifier": self.training_pipeline.model.get_architecture(),
            "random_seed": self.training_pipeline.random_seed,
            "hyperparams": self.training_pipeline.hyperparams,
            "stats": self.evaluation_pipeline.get_stats(),
        }

        if save_run_info:
            # Save metadata
            with open(paths.run_info_dir / f"run_{self.timestamp}.json", "w") as f:
                json.dump(run_info, f)

            # Save model weights
            torch.save(
                self.training_pipeline.model.state_dict(),
                paths.weights_dir / f"run_{self.timestamp}_weights.pth",
            )

            # Save plots
            self.training_pipeline.loss_figure.savefig(
                paths.plots_dir / f"run_{self.timestamp}_train_val_loss.png",
                dpi=300,
            )


class EvaluationPipeline(ABC):
    def __init__(
        self,
        model,
        test_data=None,
        language: str = "english",
        batch_size: int = 128,
        encoder: Encoder = None,
    ):
        self.model = model
        self.test_data = test_data
        self.language = language
        self.batch_size = batch_size
        self.test_loader = None
        self.encoder = encoder

    def unpack_data(self):
        if self.test_data is None:
            # Unpack data from parquet files
            self.test_data = pd.read_parquet(
                paths.embeddings_dir / self.language / self.encoder.model_name / "test.parquet"
            )

    def split_data(self):
        self.unpack_data()

        self.X_test = np.array(self.test_data["embeddings"])
        self.y_test = np.array(self.test_data["labels"])

    @abstractmethod
    def evaluate_model(self):
        pass

class EvaluationPipelineSklearn(EvaluationPipeline):
    def __init__(self, model, data):
        super().__init__(model, data)

    def compute_metrics(self, y_pred):
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred)
        self.recall = recall_score(self.y_test, y_pred)
        self.f1 = f1_score(self.y_test, y_pred)

        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1:.4f}")

    def evaluate_model(self):
        self.split_data()

        print("Evaluating sklearn model...")
        y_pred = self.model.predict(self.X_test)

        self.compute_metrics(y_pred)

    def get_stats(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def run(self):
        self.evaluate_model()


class EvaluationPipelineNN(EvaluationPipeline):
    def __init__(self, model, data, language, encoder, random_seed: int = 42):
        super().__init__(model, data, language, encoder=encoder)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.random_seed = random_seed

    def create_data_loaders(self):
        self.split_data()
        X_test = torch.Tensor(self.X_test)
        y_test = torch.Tensor(self.y_test)

        test_dataset = TensorDataset(X_test, y_test)

        g = torch.Generator()
        g.manual_seed(self.random_seed)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            generator=g,
        )

    def compute_metrics(self, y_pred):
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.precision = precision_score(self.y_test, y_pred, zero_division=0)
        self.recall = recall_score(self.y_test, y_pred, zero_division=0)
        self.f1 = f1_score(self.y_test, y_pred, zero_division=0)

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
            "f1": self.f1,
        }
        return stats

    def run(self):
        print("Running evaluation pipeline...")
        self.create_data_loaders()
        self.evaluate_model()


class MasterPipelineSklearn:
    def __init__(
        self,
        encoder: Encoder,
        classifier: Any,
        language: str,
    ):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ingestion_pipeline = IngestionPipeline(
            language=language, encoder=encoder
        )
        self.language = language
        self.classifier = classifier

    def run(self, save_run_info: bool = True):
        self.ingestion_pipeline.run()
        self.training_pipeline = TrainPipelineSklearn(
            language=self.language,
            model=self.classifier,
            data=(self.ingestion_pipeline.train_data, self.ingestion_pipeline.val_data),
        )
        self.training_pipeline.run()
        self.evaluation_pipeline = EvaluationPipelineSklearn(
            model=self.training_pipeline.model,
            data=self.ingestion_pipeline.test_data,
        )
        self.evaluation_pipeline.run()
        self.log_run()

    def log_run(self, save_run_info: bool = True):
        run_info = {
            "language": self.language,
            "encoder": self.encoder.get_params(),
            "model_type": self.training_pipeline.model.__class__.__name__,
            "random_seed": self.training_pipeline.random_seed,
            "hyperparams": self.training_pipeline.model.get_params(),
            "stats": self.evaluation_pipeline.get_stats(),
        }
        print(run_info)
    