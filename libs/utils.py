import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch

from libs.MilSurgery import MilSurgery
from libs.MilSurgeryEval import MilSurgeryEval


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(root_model, store_name, state, epoch=None):
    filename = f"{root_model}/{store_name}/ckpt.pth.tar"
    torch.save(state, filename)
    if epoch:
        filename = f"{root_model}/{store_name}/{epoch}_ckpt.pth.tar"
        torch.save(state, filename)


def prepare_loaders(train_root, val_root, cfg):

    train_dataset = MilSurgery(
        root=train_root,
        box_features_dir=cfg.data.box_features_dir,
        video_label_name=cfg.data.video_label_name,
        sample_num=cfg.data.sample_num,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
    )

    train_eval_dataset = MilSurgeryEval(
        root=train_root,
        box_features_dir=cfg.data.box_features_dir,
        frame_label_name=cfg.data.frame_label_name,
        video_label_name=cfg.data.video_label_name,
    )

    train_eval_loader = torch.utils.data.DataLoader(
        train_eval_dataset,
        batch_size=1,
    )

    val_dataset = MilSurgery(
        root=val_root,
        box_features_dir=cfg.data.box_features_dir,
        video_label_name=cfg.data.video_label_name,
        sample_num=cfg.data.sample_num,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
    )

    val_eval_dataset = MilSurgeryEval(
        root=val_root,
        box_features_dir=cfg.data.box_features_dir,
        frame_label_name=cfg.data.frame_label_name,
        video_label_name=cfg.data.video_label_name,
    )

    val_eval_loader = torch.utils.data.DataLoader(
        val_eval_dataset,
        batch_size=1,
    )

    return train_loader, val_loader, train_eval_loader, val_eval_loader


def load_pretrain_model(model, cfg):
    checkpoint = torch.load(
        os.path.join("pretrain_model", cfg.data.box_features_dir, "model_final.pth")
    )
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "roi_heads" in k:
            name = k.replace("roi_heads.", "")  # remove `roi_heads.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


def compute_pos_weight(
    root="./data/train",
    frame_label_name="video.json",
    class_num=7,
):
    frame_label_path = os.path.join(root, "annotations", frame_label_name)
    frame_label_dict = json.load(open(frame_label_path, "r"))
    all_counts = np.array([1] * class_num)
    pos_counts = np.array([0] * class_num)
    for k, v in frame_label_dict.items():
        pos_counts += np.array(v)
        all_counts += np.array([1] * class_num)
    neg_count = all_counts - pos_counts
    return torch.as_tensor(neg_count / (pos_counts + 1e-5), dtype=torch.float)
