import argparse
import json
from argparse import Namespace


MODEL_CHOICES = ["Tractomamba", "tractomamba"]


def create_parser():
    parser = argparse.ArgumentParser(description="Configure Tractomamba training or evaluation runs.")

    # Paths
    parser.add_argument("--input_path", type=str, default="./TrainData", help="Input training data directory.")
    parser.add_argument("--out_path_base", type=str, default="./ModelWeights", help="Directory for saving trained models.")

    # Direction augmentation
    parser.add_argument("--flip_aug", action="store_true", default=False, help="Enable streamline direction flip augmentation.")
    parser.add_argument("--flip_prob", type=float, default=0, help="Probability of applying flip augmentation.")

    # Spatial augmentation
    parser.add_argument("--rot_ang_lst", type=str, default="0 0 0", help='Rotation ranges for LR/AP/SI axes, for example "45 15 15".')
    parser.add_argument("--scale_ratio_range", type=str, default="0 0", help="Random scale range around 1.0, for example 0.35 0.05.")
    parser.add_argument("--trans_dis", type=float, default=0, help="Random translation range [-trans_dis, +trans_dis].")
    parser.add_argument("--aug_times", type=int, default=0, help="Number of augmented samples generated for each subject.")

    # Local-global representation
    parser.add_argument("--k", type=int, default=30, help="Number of local neighbor streamlines.")
    parser.add_argument("--k_ds_rate", type=float, default=1.0, help="Downsample rate for pairwise distance calculation; 1.0 means no downsampling.")
    parser.add_argument("--k_global", type=int, default=800, help="Number of globally sampled streamlines.")
    parser.add_argument("--k_point_level", type=int, default=5, help="Number of neighbor points on one streamline.")

    # Training parameters
    parser.add_argument("--save_step", type=int, default=5, help="Weight saving interval.")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of data loading workers.")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Embedding dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--opt", type=str, default="Adam", help="Optimizer type.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for Adam.")
    parser.add_argument("--momentum", type=float, default=0, help="Momentum for SGD.")
    parser.add_argument("--scheduler", type=str, default="step", help="Learning rate scheduler type.")
    parser.add_argument("--step_size", type=int, default=8, help="Learning rate decay period.")
    parser.add_argument("--decay_factor", type=float, default=0.5, help="Learning rate decay factor.")
    parser.add_argument("--T_0", type=int, default=10, help="First restart length for cosine scheduling.")
    parser.add_argument("--T_mult", type=int, default=2, help="Restart length multiplier.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Classifier dropout.")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=1024, help="Validation batch size.")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="Testing batch size.")
    parser.add_argument("--epoch", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--best_metric", type=str, default="f1", help="Metric used to select the best model.")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="Tractomamba", choices=MODEL_CHOICES, help="Model architecture name.")
    parser.add_argument("--mamba_layers", type=int, default=1, help="Number of Mamba/SSM layers in each Mamba stack.")
    parser.add_argument("--mamba_dropout", type=float, default=0.2, help="Dropout used inside Mamba stacks.")
    parser.add_argument("--num_fiber_per_brain", type=int, default=10000, help="Number of fibers per brain.")
    parser.add_argument("--num_point_per_fiber", type=int, default=15, help="Number of points per fiber.")

    parser.add_argument("--use_tracts_training", default=False, action="store_true", help="Convert cluster labels into tract labels during training.")
    parser.add_argument("--use_tracts_testing", default=False, action="store_true", help="Convert cluster labels into tract labels during testing.")
    parser.add_argument("--save_args_only", default=False, action="store_true", help="Save arguments only without launching training.")
    parser.add_argument("--cal_equiv_dist", default=False, action="store_true", help="Calculate equivalent distance for pairwise distance matrices.")
    parser.add_argument("--include_org_data", default=False, action="store_true", help="Include original data when using augmentation.")

    return parser


def load_args(path, args):
    params_set_in_testing = ["aug_times", "out_path"]
    with open(path, "r") as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key, value in saved_json_dict.items():
            if key in params_set_in_testing:
                print("Skip loading {} from training args".format(key))
                continue
            args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def load_args_in_testing_only(path, args):
    params_set_in_testing = ["aug_times", "out_path"]
    with open(path, "r") as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key, value in saved_json_dict.items():
            if key in params_set_in_testing:
                print("Skip loading {} from training args".format(key))
                continue
            if key in args_dict:
                args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def save_args(path, args):
    with open(path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print(args.__dict__)
