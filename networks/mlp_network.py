import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_layers_outputs = []  # To store hidden layer outputs if needed

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x, return_penultimate_features=False):
        # x is expected to be flattened: (batch_size, input_dim)
        penultimate_activation = x
        for i, layer in enumerate(self.layers):  # Iterate through hidden layers
            penultimate_activation = layer(penultimate_activation)
            penultimate_activation = F.relu(penultimate_activation)

        logits = self.output_layer(penultimate_activation)

        if return_penultimate_features:
            return logits, penultimate_activation
        else:
            return logits