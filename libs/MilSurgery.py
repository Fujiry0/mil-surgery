import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class MilSurgery(Dataset):
    def __init__(
        self,
        root="./data/train",
        box_features_dir="faster_rcnn_R_50_FPN_1x_set4",
        video_label_name="video.json",
        sample_num=32,
    ):
        self.box_features_dir = os.path.join(root, "box_features", box_features_dir)
        self.video_label_path = os.path.join(root, "annotations", video_label_name)
        self.box_feature_names = sorted(os.listdir(self.box_features_dir))
        self.video_label_dict = json.load(open(self.video_label_path, "r"))

        self.sample_num = sample_num
        self.num_all_box = 0

        # self.box_path_dict key: video num value: [(path, number of instance), ...]
        self.box_path_dict = defaultdict(list)
        for box_feature_name in self.box_feature_names:
            video_num = box_feature_name.split("_")[1]
            box_feature_path = os.path.join(self.box_features_dir, box_feature_name)
            box_dict = np.load(box_feature_path, allow_pickle=True).item()
            num_box = box_dict["box_features_"].shape[0]  # the number of box in a frame
            self.num_all_box += num_box
            self.box_path_dict[video_num].extend(
                [(box_feature_path, idx) for idx in range(num_box)]
            )

        self.video_list = list(self.box_path_dict.keys())

    def __len__(self):
        return self.num_all_box // self.sample_num

    def __getitem__(self, index):
        video_num = random.choice(self.video_list)
        sampled_box_feature_paths = random.sample(
            self.box_path_dict[video_num], self.sample_num
        )
        sampled_box_features = []
        sampled_faster_rcnn_pred_class = []
        sampled_faster_rcnn_pred_class_scores = []
        for sampled_box_feature_path in sampled_box_feature_paths:
            box_feature_path = sampled_box_feature_path[0]
            idx = sampled_box_feature_path[1]
            box_dict = np.load(box_feature_path, allow_pickle=True).item()
            box_feature = box_dict["box_features_"][idx]
            faster_rcnn_pred_class_scores = box_dict["scores_list"][idx]
            faster_rcnn_pred_class = np.zeros(len(faster_rcnn_pred_class_scores))
            faster_rcnn_pred_class[box_dict["outputs_pred_classes"][idx]] = 1
            sampled_box_features.append(box_feature)
            sampled_faster_rcnn_pred_class.append(faster_rcnn_pred_class)
            sampled_faster_rcnn_pred_class_scores.append(faster_rcnn_pred_class_scores)

        sampled_box_features = np.stack(sampled_box_features)
        sampled_faster_rcnn_pred_class = np.stack(sampled_faster_rcnn_pred_class)
        sampled_faster_rcnn_pred_class_scores = np.stack(
            sampled_faster_rcnn_pred_class_scores
        )
        video_label = np.array(self.video_label_dict[video_num])

        box_features = torch.from_numpy(sampled_box_features).type(torch.float)
        faster_rcnn_pred_class = torch.from_numpy(sampled_faster_rcnn_pred_class)
        faster_rcnn_pred_class_scores = torch.from_numpy(
            sampled_faster_rcnn_pred_class_scores
        )
        video_label = torch.from_numpy(video_label).type(torch.float)

        return (
            box_features,
            faster_rcnn_pred_class,
            faster_rcnn_pred_class_scores,
            video_label,
        )


if __name__ == "__main__":
    train_dset = MilSurgery()
    (
        box_features,
        faster_rcnn_pred_class,
        faster_rcnn_pred_class_scores,
        video_label,
    ) = train_dset[0]

    print(box_features.shape)
    print(faster_rcnn_pred_class.shape)
    print(video_label)
    print(len(train_dset))

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=16,
    )
    for _, (
        box_features,
        faster_rcnn_pred_class,
        faster_rcnn_pred_class_scores,
        video_label,
    ) in enumerate(train_loader):
        print(video_label)
        break
