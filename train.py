import json
import os
import warnings

import hydra
import torch.nn.parallel
import torch.optim
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from libs.helper import eval, train, val
from libs.MilSurgery import MilSurgery
from libs.MilSurgeryEval import MilSurgeryEval
from libs.utils import save_checkpoint, set_seed
from modeling.model import Model

warnings.simplefilter("ignore")


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: DictConfig):
    global best_f1_macro
    set_seed(cfg.seed)
    train_root = os.path.join(cfg.data.data_path, "train")
    val_root = os.path.join(cfg.data.data_path, "val")
    train_dataset = MilSurgery(
        root=train_root,
        box_features_dir=cfg.data.box_features_dir,
        video_label_name=cfg.data.video_label_name,
        sample_num=cfg.data.sample_num,
    )

    train_eval_dataset = MilSurgeryEval(
        root=train_root,
        box_features_dir=cfg.data.box_features_dir,
        frame_label_name=cfg.data.frame_label_name,
        video_label_name=cfg.data.video_label_name,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
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

    val_eval_dataset = MilSurgeryEval(
        root=val_root,
        box_features_dir=cfg.data.box_features_dir,
        frame_label_name=cfg.data.frame_label_name,
        video_label_name=cfg.data.video_label_name,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
    )

    val_eval_loader = torch.utils.data.DataLoader(
        val_eval_dataset,
        batch_size=1,
    )

    tf_writer = SummaryWriter(
        log_dir=os.path.join(cfg.logger.root_log, cfg.logger.store_name)
    )

    model = Model()
    model = model.cuda()

    if cfg.training.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.wd
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.wd
        )

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    best_f1_macro_mil = 0
    mil_outputs = {}
    for epoch in range(cfg.training.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        (
            train_f1_macro_mil,
            train_f1_macro,
            train_f1_macro_video_label,
            mil_outputs["train"],
        ) = eval(train_eval_loader, model, epoch)
        model_state = model.state_dict()

        val_loss = val(val_loader, model, criterion, epoch)
        (
            val_f1_macro_mil,
            val_f1_macro,
            val_f1_macro_video_label,
            mil_outputs["val"],
        ) = eval(val_eval_loader, model, epoch)

        scheduler.step()

        if best_f1_macro_mil < train_f1_macro_mil:
            save_checkpoint(
                cfg.logger.root_log,
                cfg.logger.store_name,
                {
                    "epoch": epoch + 1,
                    "state_dict": model_state,
                },
            )

            with open(
                os.path.join(
                    cfg.logger.root_log, cfg.logger.store_name, "predictions.json"
                ),
                "w",
            ) as f:
                json.dump(mil_outputs, f, indent=4)

            best_f1_macro_mil = train_f1_macro_mil
            print("best train f1 is updated")

        tf_writer.add_scalar("loss/train", train_loss, epoch)
        tf_writer.add_scalar("f1 mil/train", train_f1_macro_mil, epoch)
        tf_writer.add_scalar("f1 faster-rcnn pred/train", train_f1_macro, epoch)
        tf_writer.add_scalar("f1 video label/train", train_f1_macro_video_label, epoch)
        tf_writer.add_scalar("best f1 mil/train", best_f1_macro_mil, epoch)

        tf_writer.add_scalar("loss/val", val_loss, epoch)
        tf_writer.add_scalar("f1 mil/val", val_f1_macro_mil, epoch)
        tf_writer.add_scalar("f1 faster-rcnn pred/val", val_f1_macro, epoch)
        tf_writer.add_scalar("f1 video label/val", val_f1_macro_video_label, epoch)
    tf_writer.close()


if __name__ == "__main__":
    main()
