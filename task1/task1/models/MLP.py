import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, dropout: float = 0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def get_architecture(self):
        return {k: str(v) for k, v in self.named_children()}
    