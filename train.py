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
from libs.utils import (
    compute_pos_weight,
    load_pretrain_model,
    prepare_loaders,
    save_checkpoint,
    set_seed,
)
from modeling.model import Model

warnings.simplefilter("ignore")


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: DictConfig):
    global best_f1_macro
    set_seed(cfg.seed)
    train_root = os.path.join(cfg.data.data_path, "train")
    val_root = os.path.join(cfg.data.data_path, "val")

    train_loader, val_loader, train_eval_loader, val_eval_loader = prepare_loaders(
        train_root, val_root, cfg
    )

    tf_writer = SummaryWriter(
        log_dir=os.path.join(cfg.logger.root_log, cfg.logger.store_name)
    )

    box_head_fc_dims = [1024] * cfg.model.box_head_fc_dims_num

    model = Model(box_head_fc_dims=box_head_fc_dims)
    if cfg.training.use_pretrain:
        model = load_pretrain_model(model, cfg)
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

    pos_weight = None
    if cfg.training.use_pos_weight:
        pos_weight = compute_pos_weight(
            train_root,
            cfg.data.video_label_name,
        )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight).cuda()

    best_f1_macro_mil = 0
    # mil_outputs = {}
    for epoch in range(cfg.training.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        (
            train_eval_loss,
            train_f1_macro_mil_frame_preds,
            train_f1_macro_mil_frame_from_instances_preds,
            train_f1_macro_all_faster_rcnn_preds,
            train_f1_macro_all_video_label_preds,
        ) = eval(train_eval_loader, model, criterion, epoch, split="train")
        model_state = model.state_dict()

        val_loss = val(val_loader, model, criterion, epoch)
        (
            val_eval_loss,
            val_f1_macro_mil_frame_preds,
            val_f1_macro_mil_frame_from_instances_preds,
            val_f1_macro_all_faster_rcnn_preds,
            val_f1_macro_all_video_label_preds,
        ) = eval(val_eval_loader, model, criterion, epoch, split="train")

        scheduler.step()

        if best_f1_macro_mil < train_f1_macro_mil_frame_from_instances_preds:
            save_checkpoint(
                cfg.logger.root_log,
                cfg.logger.store_name,
                {
                    "epoch": epoch + 1,
                    "state_dict": model_state,
                },
            )

            # with open(
            #     os.path.join(
            #         cfg.logger.root_log, cfg.logger.store_name, "predictions.json"
            #     ),
            #     "w",
            # ) as f:
            #     json.dump(mil_outputs, f, indent=4)

            best_f1_macro_mil = train_f1_macro_mil_frame_from_instances_preds
            print("best train f1 is updated")

        # For train
        tf_writer.add_scalar("loss/train", train_loss, epoch)
        tf_writer.add_scalar("loss/val", val_loss, epoch)

        # For eval
        tf_writer.add_scalar("eval (frame) loss/train", train_eval_loss, epoch)
        tf_writer.add_scalar(
            "f1 mil frame/train", train_f1_macro_mil_frame_preds, epoch
        )
        tf_writer.add_scalar(
            "f1 mil frame from instances/train",
            train_f1_macro_mil_frame_from_instances_preds,
            epoch,
        )
        tf_writer.add_scalar(
            "f1 frame from faster-rcnn/train",
            train_f1_macro_all_faster_rcnn_preds,
            epoch,
        )
        tf_writer.add_scalar(
            "f1 frame from video label/train",
            train_f1_macro_all_video_label_preds,
            epoch,
        )
        tf_writer.add_scalar("eval (frame) loss/val", val_eval_loss, epoch)
        tf_writer.add_scalar("f1 mil frame/val", val_f1_macro_mil_frame_preds, epoch)
        tf_writer.add_scalar(
            "f1 mil frame from instances/val",
            val_f1_macro_mil_frame_from_instances_preds,
            epoch,
        )
        tf_writer.add_scalar(
            "f1 frame from faster-rcnn/val", val_f1_macro_all_faster_rcnn_preds, epoch
        )
        tf_writer.add_scalar(
            "f1 frame from video label/val", val_f1_macro_all_video_label_preds, epoch
        )
    tf_writer.close()


if __name__ == "__main__":
    main()
