import numpy as np
from sklearn.metrics import f1_score


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for _, (
        box_features,
        _,
        _,
        video_label,
    ) in enumerate(train_loader):
        if (
            not box_features.any()
        ):  # Skip when there are no faster rcnn prediction for a frame
            continue
        inputs = box_features.cuda()
        target = video_label.cuda()
        video_output, instances_output = model(inputs)
        loss = criterion(video_output, target)

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

    for _, (
        box_features,
        _,
        _,
        video_label,
    ) in enumerate(val_loader):
        if (
            not box_features.any()
        ):  # Skip when there are no faster rcnn prediction for a frame
            continue
        inputs = box_features.cuda()
        target = video_label.cuda()
        video_output, instances_output = model(inputs)
        loss = criterion(video_output, target)

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)

    print("Epoch: {} Val Loss: {:.4f}".format(epoch, epoch_loss))

    return epoch_loss


def eval(eval_loader, model, criterion, epoch, split="train"):
    model.eval()
    running_loss = 0.0
    all_mil_frame_preds = []
    all_mil_frame_from_instances_preds = []
    all_faster_rcnn_preds = []
    all_video_label_preds = []
    all_frame_target = []
    # outputs = {}
    for _, (
        _,
        frame_label,
        video_label,
        box_features,
        faster_rcnn_pred_class,
        _,
    ) in enumerate(eval_loader):
        if (
            not box_features.any()
        ):  # Skip when there are no faster rcnn prediction for a frame
            continue
        inputs = box_features.cuda()
        target = frame_label.cuda()
        frame_output, instances_output = model(inputs)
        loss = criterion(frame_output, target)
        running_loss += loss.item() * inputs.size(0)

        frame_output = frame_output.cpu().detach().numpy()
        instances_output = instances_output.cpu().detach().numpy()
        instances_output = np.argmax(instances_output, axis=1)
        instances_output_one_hot = np.zeros((instances_output.size, 7))
        instances_output_one_hot[np.arange(instances_output.size), instances_output] = 1
        frame_output_from_instances = np.any(instances_output_one_hot, axis=0)

        all_mil_frame_preds.append(frame_output)
        all_mil_frame_from_instances_preds.append(frame_output_from_instances)
        all_faster_rcnn_preds.append(faster_rcnn_pred_class.numpy())
        all_video_label_preds.append(video_label.numpy())
        all_frame_target.append(target.cpu().numpy())

        # outputs[box_feature_name[0]] = {
        #         "mil predictions": instances_output.tolist(),

        #     }

    all_mil_frame_preds = np.vstack(all_mil_frame_preds)
    all_mil_frame_from_instances_preds = np.vstack(all_mil_frame_from_instances_preds)
    all_faster_rcnn_preds = np.vstack(all_faster_rcnn_preds)
    all_video_label_preds = np.vstack(all_video_label_preds)
    all_frame_target = np.vstack(all_frame_target)

    all_mil_frame_preds = (all_mil_frame_preds > 0.5) * 1
    f1_macro_mil_frame_preds = f1_score(
        all_mil_frame_preds, all_frame_target, average="macro"
    )

    all_mil_frame_from_instances_preds = (all_mil_frame_from_instances_preds > 0.5) * 1
    f1_macro_mil_frame_from_instances_preds = f1_score(
        all_mil_frame_from_instances_preds, all_frame_target, average="macro"
    )

    all_faster_rcnn_preds = (all_faster_rcnn_preds > 0.5) * 1
    f1_macro_all_faster_rcnn_preds = f1_score(
        all_faster_rcnn_preds, all_frame_target, average="macro"
    )

    all_video_label_preds = (all_video_label_preds > 0.5) * 1
    f1_macro_all_video_label_preds = f1_score(
        all_video_label_preds, all_frame_target, average="macro"
    )

    epoch_loss = running_loss / len(eval_loader.dataset)

    if split == "val":
        print("Epoch: {} Val Loss: {:.4f}".format(epoch, epoch_loss))

    return (
        epoch_loss,
        f1_macro_mil_frame_preds,  # max instance wise output and take f1
        f1_macro_mil_frame_from_instances_preds,  # sigmoid instance output and take any and f1
        f1_macro_all_faster_rcnn_preds,  # take faster-rcnn output f1
        f1_macro_all_video_label_preds,
    )
