import torch
import torch.nn as nn
torch.manual_seed(0)

class MLP(nn.Module):
    """Configurable class for MLP models"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5, activation_function=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation_function())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation_function())
            layers.append(nn.Dropout(dropout_prob))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)