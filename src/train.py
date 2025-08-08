import copy
import logging
import os
import sys
from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path to import flow_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_model import RectifiedFlow

from hps import Hparams
from utils import linear_warmup, write_images


def preprocess_batch(args: Hparams, batch: Dict[str, Tensor], expand_pa: bool = False):
    batch["x"] = (batch["x"].to(args.device).float() - 127.5) / 127.5  # [-1, 1]
    
    # Handle both concatenated and separate parent formats
    if "pa" in batch:
        batch["pa"] = batch["pa"].to(args.device).float()
    else:
        # For separate format, move each parent to device
        for key in args.parents_x:
            if key in batch:
                batch[key] = batch[key].to(args.device).float()
    return batch


def trainer(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    writer: SummaryWriter,
    logger: logging.Logger,
):
    # Initialize RectifiedFlow instance
    rf = RectifiedFlow()
    
    for k in sorted(vars(args)):
        logger.info(f"--{k}={vars(args)[k]}")
    logger.info(f"total params: {sum(p.numel() for p in model.parameters()):,}")

    def run_epoch(dataloader: DataLoader, training: bool = True):
        model.train(training)
        model.zero_grad(set_to_none=True)
        stats = {k: 0 for k in ["loss", "mse_loss", "n"]}
        updates_skipped = 0

        mininterval = 300 if "SLURM_JOB_ID" in os.environ else 0.1
        loader = tqdm(
            enumerate(dataloader), total=len(dataloader), mininterval=mininterval
        )

        for i, batch in loader:
            batch = preprocess_batch(args, batch, expand_pa=False)
            bs = batch["x"].shape[0]
            update_stats = True

            # Prepare parents dictionary
            if args.concat_pa and "pa" in batch:
                # Handle concatenated format - need to split back
                parents = None  # You'd need to implement splitting logic
            else:
                # Handle separate format
                parents = {
                    'digit': batch.get('digit'),
                    'thickness': batch.get('thickness'), 
                    'intensity': batch.get('intensity')
                }

            if training:
                args.iter = i + 1 + (args.epoch - 1) * len(dataloader)
                
                # Sample random time steps
                t = torch.rand(bs, device=args.device)
                
                # Create flow: x_t = t*x_1 + (1-t)*x_0, where x_1 is real data, x_0 is noise
                x_t, x_0 = rf.create_flow(batch["x"], t)
                
                # Predict velocity field
                v_pred = model(x_t, t, parents)
                
                # Compute flow matching loss: ||v_pred - (x_1 - x_0)||^2
                target_v = batch["x"] - x_0
                mse_loss = rf.mse_loss(target_v, v_pred)
                
                loss = mse_loss / args.accu_steps
                loss.backward()

                if i % args.accu_steps == 0:  # gradient accumulation update
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                    writer.add_scalar("train/grad_norm", grad_norm, args.iter)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], args.iter)

                    if grad_norm < args.grad_skip:
                        optimizer.step()
                        scheduler.step()
                        ema.update()
                    else:
                        updates_skipped += 1
                        update_stats = False
                        logger.info(
                            f"Updates skipped: {updates_skipped}"
                            + f" - grad_norm: {grad_norm:.3f}"
                        )

                    model.zero_grad(set_to_none=True)

                    if args.iter % args.viz_freq == 0 or (args.iter in early_evals):
                        with torch.no_grad():
                            write_images(args, ema.ema_model, viz_batch)
            else:
                with torch.no_grad():
                    # Sample random time steps for validation
                    t = torch.rand(bs, device=args.device)
                    x_t, x_0 = rf.create_flow(batch["x"], t)
                    v_pred = ema.ema_model(x_t, t, parents)
                    target_v = batch["x"] - x_0
                    mse_loss = rf.mse_loss(target_v, v_pred)
                    loss = mse_loss

            if update_stats:
                if training:
                    loss *= args.accu_steps
                stats["n"] += bs
                stats["loss"] += loss.detach() * bs
                stats["mse_loss"] += mse_loss.detach() * bs

            split = "train" if training else "valid"
            loader.set_description(
                f' => {split} | loss: {stats["loss"] / stats["n"]:.3f}'
                + f' - mse: {stats["mse_loss"] / stats["n"]:.3f}'
                + f" - lr: {scheduler.get_last_lr()[0]:.6g}"
                + (f" - grad norm: {grad_norm:.2f}" if training else ""),
                refresh=False,
            )
        return {k: v / stats["n"] for k, v in stats.items() if k != "n"}

    viz_batch = next(iter(dataloaders["valid"]))
    n = min(16, args.bs)  # Limit visualization batch size
    viz_batch = {k: v[:n] for k, v in viz_batch.items()}
    viz_batch = preprocess_batch(args, viz_batch, expand_pa=False)
    early_evals = set([args.iter + 1] + [args.iter + 2**n for n in range(3, 14)])

    # Start training loop
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f"Epoch {args.epoch}:")

        stats = run_epoch(dataloaders["train"], training=True)

        writer.add_scalar(f"loss/train", stats["loss"], args.epoch)
        writer.add_scalar(f"mse_loss/train", stats["mse_loss"], args.epoch)
        logger.info(
            f'=> train | loss: {stats["loss"]:.4f}'
            + f' - mse_loss: {stats["mse_loss"]:.4f}'
            + f" - steps: {args.iter}"
        )

        if (args.epoch - 1) % args.eval_freq == 0:
            valid_stats = run_epoch(dataloaders["valid"], training=False)

            writer.add_scalar(f"loss/valid", valid_stats["loss"], args.epoch)
            writer.add_scalar(f"mse_loss/valid", valid_stats["mse_loss"], args.epoch)
            logger.info(
                f'=> valid | loss: {valid_stats["loss"]:.4f}'
                + f' - mse_loss: {valid_stats["mse_loss"]:.4f}'
                + f" - steps: {args.iter}"
            )

            if valid_stats["loss"] < args.best_loss:
                args.best_loss = valid_stats["loss"]
                save_dict = {
                    "epoch": args.epoch,
                    "step": args.epoch * len(dataloaders["train"]),
                    "best_loss": args.best_loss.item(),
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "hparams": vars(args),
                }
                ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
                torch.save(save_dict, ckpt_path)
                logger.info(f"Model saved: {ckpt_path}")
    return

