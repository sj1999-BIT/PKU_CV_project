
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, layer_1, layer_2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, layer_1),
            nn.Sigmoid(),
            nn.Linear(layer_1, layer_2),
            nn.Sigmoid(),
            nn.Linear(layer_2, output_dim),
            nn.Softmax(),
        )

    # Input: a vector [obj_id, obj_x, obj_y, obj_w, obj_h, subj_id, subj_x, subj_y, subj_w, subj_h, pred_id]
    # Output: probability that the relationsip between obj and subj is pred
    def forward(self, x):
        return self.network(x)
