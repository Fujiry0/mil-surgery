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
        root="./data",
        box_features_dir="faster_rcnn_R_50_FPN_1x_set4",
        video_label_name="video.json",
        sample_num=32,
    ):
        self.box_features_dir = os.path.join(root, "box_features", box_features_dir)
        self.label_path = os.path.join(root, "annotations", video_label_name)
        self.sample_num = sample_num
        self.num_all_box = 0

        self.box_feature_names = sorted(os.listdir(self.box_features_dir))
        self.box_path_dict = defaultdict(list)
        for box_feature_name in self.box_feature_names:
            video_num = box_feature_name.split("_")[1]
            box_feature_path = os.path.join(self.box_features_dir, box_feature_name)
            box_dict = np.load(box_feature_path, allow_pickle=True).item()
            num_box = (
                box_dict["box_features"][0].numpy().shape[0]
            )  # the number of box in a frame
            self.num_all_box += num_box
            self.box_path_dict[video_num].extend(
                [(box_feature_path, idx) for idx in range(num_box)]
            )

        self.video_list = list(self.box_path_dict.keys())
        self.label_dict = json.load(open(self.label_path, "r"))
        print(self.num_all_box)

    def __len__(self):
        return self.num_all_box // self.sample_num

    def __getitem__(self, index):
        video_num = random.choice(self.video_list)
        sampled_box_feature_paths = random.sample(
            self.box_path_dict[video_num], self.sample_num
        )
        sampled_box_features = []
        sampled_pred_classes = []
        for sampled_box_feature_path in sampled_box_feature_paths:
            box_feature_path = sampled_box_feature_path[0]
            idx = sampled_box_feature_path[1]
            box_dict = np.load(box_feature_path, allow_pickle=True).item()
            box_feature = box_dict["box_features"][0].numpy()[idx]
            pred_classe = box_dict["pred_instances_pred_classes"][0].numpy()[idx]
            sampled_box_features.append(box_feature)
            sampled_pred_classes.append(pred_classe)

        sampled_box_features = np.stack(sampled_box_features)
        sampled_pred_classes = np.stack(sampled_pred_classes)
        sampled_box_features = torch.from_numpy(sampled_box_features).type(torch.float)
        sampled_pred_classes = torch.from_numpy(sampled_pred_classes).type(torch.float)
        label = torch.from_numpy(np.array(self.label_dict[video_num])).type(torch.float)

        return sampled_box_features, sampled_pred_classes, label


if __name__ == "__main__":
    train_dset = MilSurgery()
    box_features, pred_classes, label = train_dset[0]
    print(box_features.shape)
    print(pred_classes.shape)
    print(label)
    print(len(train_dset))

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=16,
    )
    for _, (box_features, pred_classes, target) in enumerate(train_loader):
        print(target)
        break
