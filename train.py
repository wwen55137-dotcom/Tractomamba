import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets.dataset import unrelatedHCP_PatchData
from models.Tractomamba import TractomambaClassifier
from utils.climamba import create_parser, save_args
from utils.funcs import makepath, fix_seed
from utils.logger import create_logger


def str_to_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v) for v in value.split()]


def choose_device(device_arg):
    if device_arg == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Training Tractomamba requires a CUDA-enabled environment.")
        return torch.device("cuda")
    device = torch.device(device_arg)
    if device.type == "cpu":
        raise RuntimeError("CPU training is not supported for this Mamba-based model.")
    return device


def add_training_args(parser):
    parser.add_argument("--device", default="auto", help="auto, cuda, or cuda:N.")
    parser.add_argument("--manual_seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--save_aug_data", action="store_true", help="Save example augmented tractography files for debugging.")
    return parser


def build_dataset(args, split, logger, flip_aug):
    dataset = unrelatedHCP_PatchData(
        root=args.input_path,
        out_path=args.out_path,
        logger=logger,
        split=split,
        num_fiber_per_brain=args.num_fiber_per_brain,
        num_point_per_fiber=args.num_point_per_fiber,
        use_tracts_training=args.use_tracts_training,
        k=args.k,
        k_global=args.k_global,
        rot_ang_lst=args.rot_ang_lst,
        scale_ratio_range=args.scale_ratio_range,
        trans_dis=args.trans_dis,
        aug_times=args.aug_times,
        cal_equiv_dist=args.cal_equiv_dist,
        k_ds_rate=args.k_ds_rate,
        include_org_data=args.include_org_data,
        flip_aug=flip_aug,
        flip_prob=args.flip_prob,
    )
    dataset.save_aug_data = args.save_aug_data
    return dataset


def build_model(args, num_classes, device):
    model = TractomambaClassifier(
        k=args.k,
        k_global=args.k_global,
        num_classes=num_classes,
        feature_transform=False,
        first_feature_transform=False,
        mamba_layers=args.mamba_layers,
        mamba_dropout=args.mamba_dropout,
        dropout=args.dropout,
    )
    model.to(device)
    return model


def build_optimizer(args, model):
    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError("Please choose --opt Adam or SGD.")

    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_factor)
    elif args.scheduler == "wucd":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        raise ValueError("Please choose --scheduler step or wucd.")

    return optimizer, scheduler


def make_info_point_set(args, klocal_feat_set, global_point_set):
    klocal_feat_set = klocal_feat_set.transpose(2, 1)
    global_point_set = global_point_set.transpose(2, 1)

    if args.k == 0 and args.k_global == 0:
        return torch.zeros(1)
    if args.k == 0:
        return global_point_set
    if args.k_global == 0:
        return klocal_feat_set
    return torch.cat((klocal_feat_set, global_point_set), dim=3)


def run_epoch(model, loader, args, device, optimizer=None, scheduler=None, epoch=1):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    labels = []
    predictions = []
    num_batches = len(loader)

    for batch_idx, data in enumerate(loader):
        points, label, klocal_feat_set, global_point_set, _ = data
        label = label[:, 0].long()
        points = points.transpose(2, 1)
        info_point_set = make_info_point_set(args, klocal_feat_set, global_point_set)

        points = points.to(device)
        label = label.to(device)
        info_point_set = info_point_set.to(device)

        if is_train:
            optimizer.zero_grad()

        pred, _, _ = model(points, info_point_set)
        pred = pred.view(-1, args.num_classes)
        loss = F.nll_loss(pred, label)

        if is_train:
            loss.backward()
            optimizer.step()
            if args.scheduler == "wucd":
                scheduler.step((epoch - 1) + batch_idx / max(1, num_batches))

        total_loss += loss.item()
        predictions.extend(torch.max(pred, dim=1)[1].detach().cpu().numpy().tolist())
        labels.extend(label.detach().cpu().numpy().tolist())

    return total_loss / max(1, num_batches), np.asarray(labels), np.asarray(predictions)


def classification_metrics(labels, predictions, num_classes):
    acc = float(np.mean(labels == predictions)) if labels.size else 0.0
    f1_values = []

    for class_idx in range(num_classes):
        tp = np.sum((labels == class_idx) & (predictions == class_idx))
        fp = np.sum((labels != class_idx) & (predictions == class_idx))
        fn = np.sum((labels == class_idx) & (predictions != class_idx))
        denom = (2 * tp) + fp + fn
        f1_values.append(0.0 if denom == 0 else float((2 * tp) / denom))

    return acc, float(np.mean(f1_values))


def save_checkpoint(path, model, args, epoch, val_loss, val_acc, val_f1):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        },
        path,
    )


def main():
    parser = add_training_args(create_parser())
    args = parser.parse_args()

    args.rot_ang_lst = str_to_float_list(args.rot_ang_lst)
    args.scale_ratio_range = str_to_float_list(args.scale_ratio_range)
    args.out_path = os.path.abspath(args.out_path_base)
    args.input_path = os.path.abspath(args.input_path)
    makepath(args.out_path)

    fix_seed(args.manual_seed)
    device = choose_device(args.device)

    logger = create_logger(args.out_path)
    logger.info("=" * 55)
    logger.info(args)
    logger.info("=" * 55)

    train_dataset = build_dataset(args, "train", logger, flip_aug=args.flip_aug)
    val_dataset = build_dataset(args, "val", logger, flip_aug=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.use_tracts_training:
        args.num_classes = len(np.unique(train_dataset.org_label))
    else:
        args.num_classes = len(train_dataset.label_names)

    logger.info("Training samples: {}".format(len(train_dataset)))
    logger.info("Validation samples: {}".format(len(val_dataset)))
    logger.info("Number of classes: {}".format(args.num_classes))

    model = build_model(args, args.num_classes, device)
    optimizer, scheduler = build_optimizer(args, model)

    best_val_f1 = -1.0
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, args.epoch + 1):
        train_loss, train_labels, train_predictions = run_epoch(
            model, train_loader, args, device, optimizer=optimizer, scheduler=scheduler, epoch=epoch
        )
        if args.scheduler == "step":
            scheduler.step()

        with torch.no_grad():
            val_loss, val_labels, val_predictions = run_epoch(model, val_loader, args, device, epoch=epoch)

        train_acc, train_f1 = classification_metrics(train_labels, train_predictions, args.num_classes)
        val_acc, val_f1 = classification_metrics(val_labels, val_predictions, args.num_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        logger.info(
            "epoch [{}/{}] train loss {:.4f} acc {:.4f} f1 {:.4f}; val loss {:.4f} acc {:.4f} f1 {:.4f}".format(
                epoch, args.epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1
            )
        )

        if epoch % args.save_step == 0:
            save_checkpoint(
                os.path.join(args.out_path, "epoch_{}_checkpoint.pth".format(epoch)),
                model,
                args,
                epoch,
                val_loss,
                val_acc,
                val_f1,
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                os.path.join(args.out_path, "best_checkpoint.pth"),
                model,
                args,
                epoch,
                val_loss,
                val_acc,
                val_f1,
            )
            torch.save(best_state, os.path.join(args.out_path, "best_tractomamba_model.pth"))

    with open(os.path.join(args.out_path, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_args(os.path.join(args.out_path, "cli_args.txt"), args)

    logger.info("Best validation f1: {:.4f}".format(best_val_f1))
    logger.info("Total training time: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
