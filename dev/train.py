import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import toml

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from icecream import ic, install

install()
ic.configureOutput(includeContext=True)
ic.enable()


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def parser():
    parser = MyParser("Transformer pipeline", add_help=True)

    parser.add_argument(
        "--data_path",
        default="../data/training_cohorts/new_nacc_revised_selection.csv",
        type=str,
        help="Please specify path to the entire data.",
    )
    parser.add_argument(
        "--train_path",
        default="../data/train_vld_test_split_updated/demo_train.csv",
        type=str,
        help="Please specify path to the training data.",
    )
    parser.add_argument(
        "--vld_path",
        default="../data/train_vld_test_split_updated/demo_vld.csv",
        type=str,
        help="Please specify path to the validation data.",
    )
    parser.add_argument(
        "--test_path",
        default="../data/train_vld_test_split_updated/nacc_test_with_np_cli.csv",
        type=str,
        help="Please specify path to the testing data.",
    )
    parser.add_argument(
        "--cnf_file",
        default="./dev/data/toml_files/default_conf_new.toml",
        type=str,
        help="Please specify path to the configuration file.",
    )
    parser.add_argument("--img_mode", type=int, default=-1, choices=[-1])
    parser.add_argument(
        "--img_net",
        type=str,
        default="NonImg",
        choices=["NonImg"],
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help=(
            "Please specify the ckpt path for saving and/or loading the model. "
            "To load from this ckpt, please pass along the --load_from_ckpt flag."
        ),
    )
    parser.add_argument(
        "--load_from_ckpt",
        action="store_true",
        help="Set to True to load model from checkpoint.",
    )
    parser.add_argument(
        "--save_intermediate_ckpts",
        action="store_true",
        help="Set to True to save intermediate model checkpoints.",
    )
    parser.add_argument("--wandb", action="store_true", help="Set to True to init wandb logging.")
    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="Set to True for balanced sampling.",
    )
    parser.add_argument(
        "--ranking_loss",
        action="store_true",
        help="Set to True to apply ranking loss.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Set True for DP training.",
    )
    parser.add_argument(
        "--d_model",
        default=64,
        type=int,
        help="Please specify the dimention of the feature embedding",
    )
    parser.add_argument(
        "--nhead",
        default=1,
        type=int,
        help="Please specify the number of transformer heads",
    )
    parser.add_argument(
        "--num_epochs",
        default=256,
        type=int,
        help="Please specify the number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Please specify the batch size",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Please specify the learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=2,
        type=float,
        help="Please specify the gamma value for the focal loss",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Please specify the weight decay (optional)",
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    return args


def validate_nonimg_args(args):
    if args.img_mode != -1:
        raise ValueError("This project build only supports --img_mode -1.")
    if args.img_net != "NonImg":
        raise ValueError("This project build only supports --img_net NonImg.")


args = parser()
validate_nonimg_args(args)

print(f"Image backbone: {args.img_net}")

if args.ckpt_path:
    save_path = os.path.dirname(args.ckpt_path)
    if save_path:
        os.makedirs(save_path, exist_ok=True)


seed = 0
stripped = "_stripped_MNI"
print("Loading training dataset ... ")
dat_trn = CSVDataset(
    dat_file=args.train_path,
    cnf_file=args.cnf_file,
    mode=0,
    img_mode=args.img_mode,
    arch=args.img_net,
    transforms=None,
    stripped=stripped,
)
print("Done.\nLoading Validation dataset ...")
dat_vld = CSVDataset(
    dat_file=args.vld_path,
    cnf_file=args.cnf_file,
    mode=1,
    img_mode=args.img_mode,
    arch=args.img_net,
    transforms=None,
    stripped=stripped,
)
print("Done.\nLoading testing dataset ...")
dat_tst = CSVDataset(
    dat_file=args.test_path,
    cnf_file=args.cnf_file,
    mode=2,
    img_mode=args.img_mode,
    arch=args.img_net,
    transforms=None,
    stripped=stripped,
)

label_fractions = dat_trn.label_fractions

df = pd.read_csv(args.data_path)

label_distribution = {}
cnf = toml.load(args.cnf_file)
for label in list(cnf["label"].keys()):
    label_distribution[label] = dict(df[label].value_counts())
ckpt_path = args.ckpt_path

print(label_fractions)
print(label_distribution)

mdl = ADRDModel(
    src_modalities=dat_trn.feature_modalities,
    tgt_modalities=dat_trn.label_modalities,
    label_fractions=label_fractions,
    d_model=args.d_model,
    nhead=args.nhead,
    num_encoder_layers=1,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    batch_size_multiplier=1,
    lr=args.lr,
    weight_decay=args.weight_decay,
    gamma=args.gamma,
    criterion="AUC (ROC)",
    device="cuda",
    cuda_devices=[0],
    img_net=args.img_net,
    ckpt_path=ckpt_path,
    load_from_ckpt=args.load_from_ckpt,
    save_intermediate_ckpts=args.save_intermediate_ckpts,
    data_parallel=True,
    verbose=4,
    wandb_=args.wandb,
    label_distribution=label_distribution,
    ranking_loss=args.ranking_loss,
    _amp_enabled=False,
    _dataloader_num_workers=4,
)

mdl.fit(
    dat_trn.features,
    dat_vld.features,
    dat_trn.labels,
    dat_vld.labels,
    img_train_trans=None,
    img_vld_trans=None,
    img_mode=args.img_mode,
)
