"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import time
import csv
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA


from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
# from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    # MODIFIED FOR CUSTOM DATASET


    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    pretrained_path = "./models/weights/AASIST.pth"

    state_dict = torch.load(
        pretrained_path,
        map_location=device
    )

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Loaded pretrained model")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:

        print("Evaluation mode disabled")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)
    

    
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")


    # Training
    

    best_val_eer = 1.0
    best_val_acc = 0.0

    csv_file = model_tag / "epoch_times.csv"

    with open(csv_file, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "epoch",
                "training_minutes",
                "training_hours"
            ]
        )

    for epoch in range(config["num_epochs"]):

        epoch_start = time.time()

        print(
            "Start training epoch{:03d}".format(
                epoch
            )
        )

        running_loss = train_epoch(
            trn_loader,
            model,
            optimizer,
            device,
            scheduler,
            config
        )

        val_acc, val_precision, val_recall, val_f1, val_eer = evaluate_metrics(
            dev_loader,
            model,
            device
        )

        print(
            "Loss:{:.5f} "
            "ValAcc:{:.4f} "
            "Precision:{:.4f} "
            "Recall:{:.4f} "
            "F1:{:.4f} "
            "EER:{:.4f}".format(
                running_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
                val_eer
            )
        )

        writer.add_scalar(
            "loss",
            running_loss,
            epoch
        )

        writer.add_scalar(
            "val_acc",
            val_acc,
            epoch
        )

        writer.add_scalar(
            "val_precision",
            val_precision,
            epoch
        )

        writer.add_scalar(
            "val_recall",
            val_recall,
            epoch
        )

        writer.add_scalar(
            "val_f1",
            val_f1,
            epoch
        )

        writer.add_scalar(
            "val_eer",
            val_eer,
            epoch
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_eer < best_val_eer:
            best_val_eer = val_eer

            writer.add_scalar(
                "best_val_eer",
                best_val_eer,
                epoch
            )

            print(
                "Best model found at epoch {} (EER={:.4f}, Acc={:.4f})".format(
                    epoch,
                    val_eer,
                    val_acc
                )
            )

            torch.save(
                model.state_dict(),
                model_save_path / "best.pth"
            )

        print(
            "Saving epoch {} for swa".format(
                epoch
            )
        )

        optimizer_swa.update_swa()
        n_swa_update += 1

        epoch_end = time.time()

        epoch_seconds = epoch_end - epoch_start

        epoch_minutes = epoch_seconds / 60

        epoch_hours = epoch_seconds / 3600

        print(
            "Epoch {:03d} took {:.2f} min ({:.3f} hr)".format(
                epoch,
                epoch_minutes,
                epoch_hours
            )
        )

        with open(csv_file, "a", newline="") as f:
            writer_csv = csv.writer(f)

            writer_csv.writerow(
                [
                    epoch,
                    round(epoch_minutes, 2),
                    round(epoch_hours, 4)
                ]
            )


    # ==========================================
    # Final Test Evaluation
    # ==========================================

    print("Start final evaluation")

    if n_swa_update > 0:

        optimizer_swa.swap_swa_sgd()

        optimizer_swa.bn_update(
            trn_loader,
            model,
            device=device
        )

    test_acc, test_precision, test_recall, test_f1, test_eer = evaluate_metrics(
        eval_loader,
        model,
        device
    )

    print(
        "\nTEST RESULTS\n"
        "Accuracy : {:.4f}\n"
        "Precision: {:.4f}\n"
        "Recall   : {:.4f}\n"
        "F1       : {:.4f}\n"
        "EER      : {:.4f}".format(
            test_acc,
            test_precision,
            test_recall,
            test_f1,
            test_eer
        )
    )

    f_log = open(
        model_tag / "metric_log.txt",
        "a"
    )

    f_log.write("=" * 5 + "\n")
    f_log.write(
        "Best Val Accuracy: {:.4f}\n".format(
            best_val_acc
        )
    )
    f_log.write(
        "Best Val EER: {:.4f}\n".format(
            best_val_eer
        )
    )
    f_log.write(
    "Test Accuracy : {:.4f}\n".format(test_acc)
    )

    f_log.write(
        "Test Precision: {:.4f}\n".format(test_precision)
    )

    f_log.write(
        "Test Recall   : {:.4f}\n".format(test_recall)
    )

    f_log.write(
        "Test F1       : {:.4f}\n".format(test_f1)
    )

    f_log.write(
        "Test EER      : {:.4f}\n".format(test_eer)
    )

    f_log.close()

    torch.save(
        model.state_dict(),
        model_save_path / "swa.pth"
    )

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / development / evaluation"""

    # ==========================================================
    # MODIFIED FOR CUSTOM DATASET
    # Instead of ASVspoof protocol files, use CSV metadata files
    # ==========================================================

    trn_csv = database_path / "train.csv"
    dev_csv = database_path / "eval.csv"
    eval_csv = database_path / "test.csv"

    # =========================
    # Training Set
    # =========================

    d_label_trn, file_train = genSpoof_list(trn_csv)

    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=None  # MODIFIED: no longer used
    )

    gen = torch.Generator()
    gen.manual_seed(seed)

    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=gen
    )

    # =========================
    # Validation Set
    # =========================

    d_label_dev, file_dev = genSpoof_list(dev_csv)

    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_dev,
        labels=d_label_dev
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # =========================
    # Test Set
    # =========================

    d_label_eval, file_eval = genSpoof_list(eval_csv)

    print("no. evaluation files:", len(file_eval))

    eval_set = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_eval,
        labels=d_label_eval
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    return trn_loader, dev_loader, eval_loader



def evaluate_metrics(
    data_loader,
    model,
    device
):

    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():

        for batch_x, batch_y in data_loader:

            batch_x = batch_x.to(device)

            batch_y = (
                batch_y
                .view(-1)
                .type(torch.int64)
                .to(device)
            )

            _, outputs = model(batch_x)

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            preds = torch.argmax(
                outputs,
                dim=1
            )

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score
    )

    eer = brentq(
        lambda x: 1. - x - interp1d(
            fpr,
            tpr
        )(x),
        0.,
        1.
    )

    return (
        acc,
        precision,
        recall,
        f1,
        eer
    )


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())