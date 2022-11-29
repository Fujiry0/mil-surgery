from typing import Dict, List

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from torch import nn

# Extract from the https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/box_head.py


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        input_shape: Dict,
        fc_dims: List[int],
    ):
        super().__init__()
        self._output_size = (
            input_shape["channels"],
            input_shape["width"],
            input_shape["height"],
        )

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


if __name__ == "__main__":
    input_shape = {"channels": 256, "height": 7, "width": 7}
    fc_dims = [1024, 1024]
    cuda = torch.device("cuda")
    model = FastRCNNConvFCHead(input_shape, fc_dims).cuda()
    box_features = torch.randn(([1000, 256, 7, 7]), device=cuda)
    output = model(box_features)
    print(f"Input shape: {box_features.shape}")
    print(f"Output shape: {output.shape}")
