import torch
from torch import nn, optim
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from utils.misc import log, save_model
from utils.trainer import evaluate
import os
import time
from tqdm import tqdm
import math

def distill_single_batch(student: nn.Module, teacher: nn.Module, alpha: float, T: float, data: torch.Tensor, targets: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable, device: torch.device) -> Tuple[float, int]:
    """Performs a single distillation step.

    Args:
        student (nn.Module): Model instance.
        teacher (nn.Module): Frozen teacher model.
        alpha (float): Weight parameter.
        T (float): Temperature for KDLoss.
        data (torch.Tensor): Data tensor, of shape (batch_size, dim_1, ... , dim_N).
        targets (torch.Tensor): Target tensor, of shape (batch_size).
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        device (torch.device): Device.

    Returns:
        float: Loss scalar.
        int: Number of correct preds.
    """

    data, targets = data.to(device), targets.to(device)

    with torch.no_grad():
        teacher_preds = teacher(data)

    optimizer.zero_grad()
    student_preds = student(data)
    loss = criterion(student_preds, targets, teacher_preds, alpha, T)
    loss.backward()
    optimizer.step()

    correct = student_preds.argmax(1).eq(targets).sum()
    return loss.item(), correct.item()


def update_temperature(step: int, max_steps: int, max_T: float, min_T: float = 1.0):
    """Uses a cosine annealing formula for calculating temperature parameter.

    Args:
        step (int): Current step.
        max_steps (int): Maximum steps for which temperature will be applied.
        max_T (float): Maximum temperature value.
        min_T (float, optional): Minimum temperature value. Defaults to 1.0. (At 1, T has no effect on softmax)

    Returns:
        _type_: _description_
    """
    if step >= max_steps:
        return min_T
    else:
        cos_val = math.cos((math.pi * (step % max_steps)) / max_steps)
        return max((max_T / 2) * (cos_val + 1), min_T)


def distill(student: nn.Module, teacher:nn.Module, optimizer: optim.Optimizer, criterion: Callable, trainloader: DataLoader, valloader: DataLoader, schedulers: dict, config: dict) -> None:
    """Trains model.

    Args:
        student (nn.Module): Model instance.
        teacher (nn.Module): Frozen teacher model.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        schedulers (dict): Dict containing schedulers.
        config (dict): Config dict.
    """
    
    step = 0
    best_acc = 0.0
    n_batches = len(trainloader)
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    
    ############################
    # start distill training
    ############################

    distill_params = config["hparams"]["distill"]
    alpha = distill_params["alpha"]
    max_T = distill_params["max_T"]
    T_epochs = distill_params["T_epochs"]

    student.train()
    teacher.eval()
    
    criterion_eval = nn.CrossEntropyLoss()

    for epoch in range(config["hparams"]["start_epoch"], config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        correct = 0

        for data, targets in trainloader:

            if schedulers["warmup"] != None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] != None and epoch < config["hparams"]["scheduler"]["max_epochs"]:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################

            T = update_temperature(step, T_epochs * n_batches, max_T)
            loss, corr = distill_single_batch(student, teacher, alpha, T, data, targets, optimizer, criterion, device)
            running_loss += loss
            correct += corr

            if not step % config["exp"]["log_freq"]:       
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"], "T": T}
                log(log_dict, step, config)

            step += 1
            
        #######################
        # epoch complete
        #######################

        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0, "train_acc": correct/(len(trainloader.dataset)), "avg_loss_per_ep": running_loss/len(trainloader)}
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            val_acc, avg_val_loss = evaluate(student, criterion_eval, valloader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)

            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, student, optimizer, schedulers["scheduler"], log_file) 

    ###########################
    # training complete
    ###########################

    val_acc, avg_val_loss = evaluate(student, criterion_eval, valloader, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config)

    # save final ckpt
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, val_acc, save_path, student, optimizer, schedulers["scheduler"], log_file)