from pipelines.TrainPipeline import TrainPipeline
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class TrainPipelineNN(TrainPipeline):
    def __init__(self, model, data_path, data, batch_size=128, model_hyperparams: dict = None):
        super().__init__(model, data_path, data, batch_size)
        self.hyperparams = model_hyperparams
        self.train_losses = []
        self.val_losses = []
        
    def create_data_loaders(self):
        """Create PyTorch DataLoaders for training and validation."""
        self.split_data()
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
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        epochs = self.hyperparams["epochs"]
        lr = self.hyperparams["lr"]
        weight_decay = self.hyperparams["weight_decay"]
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        print(f"Training running on: {self.device}")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self.validate(self.val_loader, loss_fn)
            
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
