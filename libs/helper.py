import numpy as np
import torch
from sklearn.metrics import f1_score


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for _, (inputs, _, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        target = target.cuda()
        video_level_output, _ = model(inputs)
        loss = criterion(video_level_output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    print("Epoch: {} Train Loss: {:.4f}".format(epoch, epoch_loss))

    return epoch_loss


def val(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0

    for _, (inputs, _, target) in enumerate(val_loader):
        inputs = inputs.cuda()
        target = target.cuda()
        video_level_output, _ = model(inputs)
        loss = criterion(video_level_output, target)

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)

    print("Epoch: {} Val Loss: {:.4f}".format(epoch, epoch_loss))

    return epoch_loss


def eval(eval_loader, model, epoch):
    model.eval()
    all_preds = []
    all_preds_mil = []
    all_targets = []
    all_video_target = []
    mil_outputs = {}
    with torch.no_grad():
        for _, (
            box_feature_name,
            inputs,
            pred_classes,
            frame_level_label,
            frame_level_pred_label,
            video_level_label,
        ) in enumerate(eval_loader):
            if not inputs.any():
                continue
            inputs = inputs.cuda()
            _, frame_level_output = model(inputs)
            frame_level_output = frame_level_output.cpu().numpy()
            mil_outputs[box_feature_name[0]] = {
                "mil predictions": frame_level_output.tolist(),
                "faster-rcnn prediction": pred_classes.numpy().tolist(),
            }
            frame_level_output = np.argmax(frame_level_output, axis=1)
            frame_level_output_one_hot = np.zeros((frame_level_output.size, 7))
            frame_level_output_one_hot[
                np.arange(frame_level_output.size), frame_level_output
            ] = 1
            frame_level_output = np.any(frame_level_output_one_hot, axis=0)
            all_targets.append(frame_level_label.numpy())
            all_preds_mil.append(frame_level_output)
            all_preds.append(frame_level_pred_label.cpu().numpy())
            all_video_target.append(video_level_label.numpy())
        all_targets = np.vstack(all_targets)
        all_preds_mil = np.vstack(all_preds_mil)
        all_preds = np.vstack(all_preds)
        all_video_target = np.vstack(all_video_target)

        all_preds_mil = (all_preds_mil > 0.5) * 1
        f1_macro_mil = f1_score(all_preds_mil, all_targets, average="macro")

        all_preds = (all_preds > 0.5) * 1
        f1_macro = f1_score(all_preds, all_targets, average="macro")

        all_video_target = (all_video_target > 0.5) * 1
        f1_macro_video_label = f1_score(all_video_target, all_targets, average="macro")

        return f1_macro_mil, f1_macro, f1_macro_video_label, mil_outputs
