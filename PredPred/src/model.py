
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim,  layer_1, layer_2, layer_3, output_dim, p = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer_1),
            nn.BatchNorm1d(layer_1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(layer_1, layer_2),
            nn.BatchNorm1d(layer_2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(layer_2, layer_3),
            nn.BatchNorm1d(layer_3),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(layer_3, output_dim),
        )

    def forward(self, x):
        return self.network(x)
