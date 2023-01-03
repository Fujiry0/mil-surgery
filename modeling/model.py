import torch
from torch import nn

from modeling.roi_heads.box_head import FastRCNNConvFCHead
from modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class Model(nn.Module):
    def __init__(
        self,
        box_head_input_shape={"channels": 256, "height": 7, "width": 7},
        box_head_fc_dims=[1024, 1024],
        box_predictor_input_shape={"channels": 1024, "height": None, "width": None},
        num_classes=7,
    ):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.box_head = FastRCNNConvFCHead(
            input_shape=box_head_input_shape, fc_dims=box_head_fc_dims
        )
        self.box_predictor = FastRCNNOutputLayers(
            input_shape=box_predictor_input_shape, num_classes=self.num_classes
        )

    def forward(self, box_features):
        B, N, C, H, W = box_features.shape
        box_features = box_features.view(-1, C, H, W)
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # ignore the background class
        predictions = predictions[:, : self.num_classes]
        # for computing the frame level f1 score
        frame_level_predictions = nn.Softmax(dim=1)(predictions)
        predictions = predictions.view(B, N, self.num_classes)
        predictions = torch.max(predictions, dim=1)[0]

        return predictions, frame_level_predictions


if __name__ == "__main__":
    cuda = torch.device("cuda")
    model = Model().cuda()
    batch_size = 1
    sample_num = 4
    box_features = torch.randn(([batch_size, sample_num, 256, 7, 7]), device=cuda)
    output, output_val = model(box_features)
    print(f"Input shape: {box_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output for val shape: {output_val.shape}")
    print(output)
    print(output_val)
