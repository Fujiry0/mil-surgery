import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MilSurgeryEval(Dataset):
    def __init__(
        self,
        root="./data/train",
        box_features_dir="faster_rcnn_R_50_FPN_1x_set4",
        frame_label_name="frame.json",
        video_label_name="video.json",
    ):
        self.box_features_dir = os.path.join(root, "box_features", box_features_dir)
        self.frame_label_path = os.path.join(root, "annotations", frame_label_name)
        self.video_label_path = os.path.join(root, "annotations", video_label_name)
        self.box_feature_names = sorted(os.listdir(self.box_features_dir))
        self.frame_label_dict = json.load(open(self.frame_label_path, "r"))
        self.video_label_dict = json.load(open(self.video_label_path, "r"))

    def __len__(self):
        return len(self.box_feature_names)

    def __getitem__(self, index):
        box_feature_path = os.path.join(
            self.box_features_dir, self.box_feature_names[index]
        )
        video_num = self.box_feature_names[index].split("_")[1]
        box_dict = np.load(box_feature_path, allow_pickle=True).item()
        box_features = box_dict["box_features"][0].numpy()
        pred_classes = box_dict["pred_instances_pred_classes"][0].numpy()

        box_features = torch.from_numpy(box_features)
        box_feature_name = self.box_feature_names[index].split(".")[0]

        frame_level_label = torch.from_numpy(
            np.array(self.frame_label_dict[box_feature_name + ".png"])
        )
        frame_level_pred_label = np.array(
            [1 if i in pred_classes else 0 for i in range(len(frame_level_label))]
        )

        pred_classes = torch.from_numpy(pred_classes)
        frame_level_pred_label = torch.from_numpy(frame_level_pred_label)

        video_level_label = torch.from_numpy(
            np.array(self.video_label_dict[video_num])
        ).type(torch.float)

        return (
            box_feature_name,
            box_features,
            pred_classes,
            frame_level_label,
            frame_level_pred_label,
            video_level_label,
        )


if __name__ == "__main__":
    train_dset = MilSurgeryEval()
    (
        box_feature_name,
        box_features,
        pred_classes,
        frame_level_label,
        frame_level_pred_label,
        video_level_label,
    ) = train_dset[301]
    print(box_feature_name)
    print(box_features.shape)
    print(pred_classes.shape)
    print(frame_level_label)
    print(frame_level_pred_label)
    print(video_level_label)
    print(len(train_dset))
