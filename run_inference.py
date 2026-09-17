import argparse
import json
import os
import sys
import time
from argparse import Namespace

import torch
import torch.utils.data
import whitematteranalysis as wma


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHT_PATH = os.path.join(ROOT, "trainedmodel", "tractomamba_pretrained.pth")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets.dataset import RealData_PatchData
from models.Tractomamba import TractomambaClassifier
from utils.funcs import (
    clusters_to_tract_labels,
    load_tract_cluster_mapping,
    makepath,
    write_tract_parcellation,
)
from utils.logger import create_logger
import utils.tract_feat as tract_feat


def load_model_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model config file not found: {path}. Use --config_path to point to a JSON config exported with the model architecture settings.")
    with open(path, "r") as f:
        return Namespace(**json.load(f))


def build_model(args, device):
    valid_model_names = {"Tractomamba", "tractomamba"}
    if args.model_name not in valid_model_names:
        raise ValueError(f"This package contains the Tractomamba architecture, got {args.model_name!r}.")

    model = TractomambaClassifier(
        k=args.k,
        k_global=args.k_global,
        num_classes=args.num_classes,
        feature_transform=False,
        first_feature_transform=False,
        mamba_layers=args.mamba_layers,
        mamba_dropout=args.mamba_dropout,
        dropout=args.dropout,
    )
    if not os.path.isfile(args.weight_path):
        raise FileNotFoundError(
            f"Model weight file not found: {args.weight_path}. "
            "Use --weight_path to point to a Tractomamba checkpoint."
        )
    weight = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    return model


def forward_batch(data, model, args, device):
    points, klocal_feat_set, global_point_set = data
    points = points.transpose(2, 1)
    klocal_feat_set = klocal_feat_set.transpose(2, 1)
    kglobal_point_set = global_point_set.transpose(2, 1)

    if args.k == 0 and args.k_global == 0:
        info_point_set = torch.zeros(1)
    elif args.k == 0:
        info_point_set = kglobal_point_set
    elif args.k_global == 0:
        info_point_set = klocal_feat_set
    else:
        info_point_set = torch.cat((klocal_feat_set, kglobal_point_set), dim=3)

    points = points.to(device)
    info_point_set = info_point_set.to(device)
    pred, _, _ = model(points, info_point_set)
    pred = pred.view(-1, args.num_classes)
    return torch.max(pred, dim=1)[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Tractomamba inference on a VTP/VTK tractography file.")
    parser.add_argument("--tractography_path", required=True, help="Input tractography .vtp/.vtk file.")
    parser.add_argument("--out_path", default=os.path.join(ROOT, "outputs"), help="Output directory.")
    parser.add_argument("--weight_path", default=DEFAULT_WEIGHT_PATH, help="Path to a Tractomamba checkpoint (.pth).")
    parser.add_argument("--config_path", default=os.path.join(ROOT, "configs", "model_config.json"), help="JSON file containing the model architecture settings.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Inference batch size.")
    parser.add_argument("--k_ds_rate", type=float, default=0.1, help="Downsample rate for local-neighbor calculation.")
    parser.add_argument("--num_classes", type=int, default=1600, help="Number of cluster/outlier classes.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu.")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. mamba_ssm in this environment requires CUDA for this model.")
        return torch.device("cuda")
    device = torch.device(device_arg)
    if device.type == "cpu":
        raise RuntimeError("CPU inference is not supported by the installed mamba_ssm CUDA kernels.")
    return device


def main():
    start_time = time.time()
    cli = parse_args()
    device = choose_device(cli.device)

    model_config = load_model_config(cli.config_path)
    model_config.weight_path = cli.weight_path
    model_config.num_classes = cli.num_classes
    model_config.num_points = getattr(model_config, "num_point_per_fiber", 15)
    model_config.inference_batch_size = cli.batch_size
    model_config.k_ds_rate = cli.k_ds_rate
    model_config.out_path = cli.out_path

    makepath(model_config.out_path)
    log_path = os.path.join(model_config.out_path, "log")
    makepath(log_path)
    logger = create_logger(log_path)
    logger.info("=" * 55)
    logger.info(model_config)
    logger.info("=" * 55)

    ordered_mapping = load_tract_cluster_mapping()
    tract_label_names = list(ordered_mapping.keys())

    pd_tractography = wma.io.read_polydata(cli.tractography_path)
    logger.info("Finish reading tractography from: {}".format(cli.tractography_path))

    feat_RAS, _ = tract_feat.feat_RAS(pd_tractography, number_of_points=model_config.num_points)
    logger.info("The number of fibers in test tractography is {}".format(feat_RAS.shape[0]))

    dataset = RealData_PatchData(
        feat_RAS,
        k=model_config.k,
        k_global=model_config.k_global,
        cal_equiv_dist=model_config.cal_equiv_dist,
        use_endpoints_dist=False,
        rough_num_fiber_each_iter=model_config.num_fiber_per_brain,
        k_ds_rate=model_config.k_ds_rate,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=model_config.inference_batch_size, shuffle=False)
    logger.info("calculating knn+random features is done")

    model = build_model(model_config, device)
    predicted = []
    with torch.no_grad():
        for data in loader:
            pred_idx = forward_batch(data, model, model_config, device)
            predicted.extend(pred_idx.cpu().numpy().tolist())

    tract_predicted = clusters_to_tract_labels(predicted, ordered_mapping)
    write_tract_parcellation(model_config, pd_tractography, tract_predicted, tract_label_names)

    logger.info("All done!!! Total time is {}s".format(round(time.time() - start_time, 3)))


if __name__ == "__main__":
    main()
