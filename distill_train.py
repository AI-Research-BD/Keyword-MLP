from argparse import ArgumentParser
from config_parser import get_config

from utils.loss import KDLoss
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.distiller import distill
from utils.trainer import evaluate
from utils.dataset import get_loader
from utils.misc import seed_everything, count_params, get_model, calc_step, log, set_grad_state

import torch
from torch import nn
import numpy as np
import wandb

import os
import yaml
import random
import time


def training_pipeline(config):
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    

    #####################################
    # initialize training items
    #####################################

    # data
    with open(config["train_list_file"], "r") as f:
        train_list = f.read().rstrip().split("\n")
    
    with open(config["val_list_file"], "r") as f:
        val_list = f.read().rstrip().split("\n")

    with open(config["test_list_file"], "r") as f:
        test_list = f.read().rstrip().split("\n")

    
    trainloader = get_loader(train_list, config, train=True)
    valloader = get_loader(val_list, config, train=False)
    testloader = get_loader(test_list, config, train=False)

    ###########
    # student
    ###########

    student = get_model(config["hparams"]["model"])
    student = student.to(config["hparams"]["device"])
    print(f"Created student model with {count_params(student)} parameters.")

    ###########
    # teacher
    ###########

    teacher = get_model(config["hparams"]["distill"]["model"])
    teacher = teacher.to(config["hparams"]["device"])
    print(f"Created teacher model with {count_params(teacher)} parameters.")

    teacher.load_state_dict(torch.load(config["hparams"]["distill"]["ckpt"])["model_state_dict"])
    set_grad_state(teacher, False)
    teacher.eval()
    print("Restored and froze teacher weights.")


    # loss
    criterion = KDLoss(num_classes=config["hparams"]["model"]["num_classes"])

    # optimizer
    optimizer = get_optimizer(student, config["hparams"]["optimizer"])
    
    # scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }

    scheduler_params = config["hparams"]["scheduler"]

    if scheduler_params["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(trainloader) * scheduler_params["n_warmup"])

    if scheduler_params["scheduler_type"] != None:
        total_iters = len(trainloader) * max(1, (scheduler_params["max_epochs"] - scheduler_params["n_warmup"]))
        schedulers["scheduler"] = get_scheduler(optimizer, scheduler_params["scheduler_type"], total_iters)
    

    #####################################
    # Resume run
    #####################################

    if config["hparams"]["restore_ckpt"]:
        ckpt = torch.load(config["hparams"]["restore_ckpt"])

        config["hparams"]["start_epoch"] = ckpt["epoch"] + 1
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        if schedulers["scheduler"]:
            schedulers["scheduler"].load_state_dict(ckpt["scheduler_state_dict"])
        
        print(f'Restored state from {config["hparams"]["restore_ckpt"]} successfully.')


    #####################################
    # Training
    #####################################

    print("Initiating distillation training.")
    distill(student, teacher, optimizer, criterion, trainloader, valloader, schedulers, config)


    #####################################
    # Final Test
    #####################################

    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1)

    criterion_eval = nn.CrossEntropyLoss()
    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(student, criterion_eval, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_acc_last": test_acc
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    student.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_acc, test_loss = evaluate(student, criterion_eval, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_acc_best": test_acc
    }
    log(log_dict, final_step, config)


def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])


    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        
        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")
        
        else:
            wandb.login()
        
        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"]):
            training_pipeline(config)
    
    else:
        training_pipeline(config)



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)