from typing import Dict

import torch
from torch import nn

# Extracted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    def __init__(
        self,
        input_shape: Dict,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        input_size = (
            input_shape["channels"]
            * (input_shape["width"] or 1)
            * (input_shape["height"] or 1)
        )
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = num_classes
        box_dim = 4
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        # for l in [self.cls_score, self.bbox_pred]:
        #     nn.init.constant_(l.bias, 0)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        # proposal_deltas = self.bbox_pred(x)
        return scores


if __name__ == "__main__":
    input_shape = {"channels": 1024, "height": None, "width": None}
    num_classes = 7
    cuda = torch.device("cuda")
    model = FastRCNNOutputLayers(input_shape, num_classes).cuda()
    box_features = torch.randn(([1000, 1024]), device=cuda)
    output = model(box_features)
    print(f"Input shape: {box_features.shape}")
    print(f"Output shape: {output.shape}")
