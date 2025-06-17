import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class NeuralNetwork(nn.Module, ABC):
    def __init__(self, in_features: int, out_features: int = 1, dropout: float = 0.3):
        super(NeuralNetwork, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_architecture(self):
        return {k: str(v) for k, v in self.named_children()}
        

class MLP(NeuralNetwork):
    def __init__(self, in_features: int, out_features: int = 1, dropout: float = 0.4):
        super(MLP, self).__init__(in_features, out_features, dropout)
        self.fc1 = nn.Linear(in_features=in_features, out_features=384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        return x


class GRU(NeuralNetwork):
    def __init__(self, in_features: int, out_features: int = 1, dropout: float = 0.3, bidirectional: bool = False):
        super(GRU, self).__init__(in_features, out_features, dropout)
        self.bidirectional = bidirectional
        self.hidden_size = 128
        self.gru = nn.GRU(
            in_features, 
            self.hidden_size, 
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        fc_input_size = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.fc = nn.Linear(fc_input_size, out_features)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        gru_out, _ = self.gru(x)
        
        if self.bidirectional:
            last_forward = gru_out[:, -1, :self.hidden_size]
            last_backward = gru_out[:, 0, self.hidden_size:]
            last_hidden = torch.cat((last_forward, last_backward), dim=1)
        else:
            last_hidden = gru_out[:, -1, :]
        
        out = self.fc(last_hidden)
        return out
