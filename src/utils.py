import copy
import os
import random
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn

from hps import Hparams


def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f


def beta_anneal(beta, step, anneal_steps):
    return min(beta, (max(1e-11, step) / anneal_steps) ** 2)


def normalize(x, x_min=None, x_max=None, zero_one=False):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    print(f"max: {x_max}, min: {x_min}")
    x = (x - x_min) / (x_max - x_min)  # [0,1]
    return x if zero_one else 2 * x - 1  # else [-1,1]


def log_standardize(x):
    log_x = torch.log(x.clamp(min=1e-12))
    return (log_x - log_x.mean()) / log_x.std().clamp(min=1e-12)  # mean=0, std=1


def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


def write_images(args: Hparams, model: nn.Module, batch: Dict[str, Tensor]):
    """Write visualization images for flow matching model"""
    try:
        from flow_model_2 import RectifiedFlow  # Use flow_model_2 since that's the current version
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from flow_model_2 import RectifiedFlow
    
    bs, c, h, w = batch["x"].shape
    
    # Postprocess function to convert tensors to numpy images
    def postprocess(x: Tensor):
        x = (x.permute(0, 2, 3, 1) + 1.0) * 127.5  # channels last, [0,255]
        return x.detach().cpu().numpy().astype(np.uint8)

    # Original images
    orig = postprocess(batch["x"])
    viz_images = [orig]

    # Generate samples using flow matching
    rf = RectifiedFlow()
    
    # Get parents from batch (should be [batch, 12] when concat_pa=True)
    parents = batch.get("pa", None)
    
    with torch.no_grad():
        # Limit samples to 8 for visualization
        num_viz = min(bs, 8)
        
        # Generate conditional samples if we have parents
        if parents is not None:
            # Create a copy of args for sampling
            sample_args = copy.deepcopy(args)
            sample_args.bs = num_viz
            
            # Use only the first num_viz samples for visualization
            viz_parents = parents[:num_viz]  # Shape: [num_viz, 12]
            
            cond_samples = model.sample(
                rf=rf,
                ars=sample_args,
                parents=viz_parents
            )
            cond_viz = postprocess(cond_samples)
            viz_images.append(cond_viz)

    # Save visualization to disk
    try:
        import imageio
        import os
        
        # Ensure save directory exists
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Trim all images to same number for visualization
        max_samples = min(num_viz, min(img.shape[0] for img in viz_images))
        viz_images = [img[:max_samples] for img in viz_images]
        
        # Stack images: [num_image_types, batch_size, h, w, c]
        # Then reshape to grid: [num_image_types * h, batch_size * w, c]
        n_types = len(viz_images)
        h, w, c = viz_images[0].shape[1:]
        
        # Create grid
        grid = np.concatenate(viz_images, axis=0)  # [total_samples, h, w, c]
        grid = grid.reshape(n_types, max_samples, h, w, c)
        grid = grid.transpose(0, 2, 1, 3, 4)  # [n_types, h, max_samples, w, c]
        grid = grid.reshape(n_types * h, max_samples * w, c)
        
        # Save image
        imageio.imwrite(os.path.join(args.save_dir, f"flow_viz-{args.iter}.png"), grid)
        
    except (ImportError, Exception) as e:
        print(f"Failed to save visualization: {e}")
    
    return viz_images