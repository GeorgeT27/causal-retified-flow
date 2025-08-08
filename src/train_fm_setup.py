import logging
import os
from typing import Any, Dict, Tuple

import send2trash
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import morphomnist
from hps import Hparams
from utils import linear_warmup, seed_worker


def setup_dataloaders(args: Hparams) -> Dict[str, DataLoader]:
    if "morphomnist" in args.hps:
        datasets = morphomnist(args)
    else:
        NotImplementedError

    kwargs = {
        "batch_size": args.bs,
        "num_workers": os.cpu_count() // 2,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **kwargs),
        "valid": DataLoader(datasets["valid"], shuffle=False, **kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **kwargs),
    }
    return dataloaders


def setup_optimizer(
    args: Hparams, model: nn.Module
) -> Tuple[torch.optim.Optimizer, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=args.betas
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)
    )

    return optimizer, scheduler


def setup_directories(args: Hparams, ckpt_dir: str = "../checkpoints") -> str:
    parents_folder = "_".join([k[0] for k in args.parents_x])
    save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    if os.path.isdir(save_dir):
        if (
            input(f"\nSave directory '{save_dir}' already exists, overwrite? [y/N]: ")
            == "y"
        ):
            if input(f"Send '{save_dir}', to Trash? [y/N]: ") == "y":
                send2trash.send2trash(save_dir)
                print("Done.\n")
            else:
                exit()
        else:
            if (
                input(f"\nResume training with save directory '{save_dir}'? [y/N]: ")
                == "y"
            ):
                pass
            else:
                exit()
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_tensorboard(args: Hparams, model: nn.Module) -> SummaryWriter:
    """Setup metric summary writer."""
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))

    # Setup custom scalars for flow matching model
    if "deeperunet" in type(model).__name__.lower():
        writer.add_custom_scalars(
            {
                "loss": {"loss": ["Multiline", ["loss/train", "loss/valid"]]},
                "mse_loss": {"mse_loss": ["Multiline", ["mse_loss/train", "mse_loss/valid"]]},
                "learning_rate": {"lr": ["Multiline", ["lr/train"]]},
                "flow_quality": {"flow_quality": ["Multiline", ["flow_quality/train", "flow_quality/valid"]]}
            }
        )
    return writer


def setup_logging(args: Hparams) -> logging.Logger:
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "trainlog.txt")),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger