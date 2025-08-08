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
        from flow_model import RectifiedFlow
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from flow_model import RectifiedFlow
    
    bs, c, h, w = batch["x"].shape
    # original imgs, channels last, [0,255]
    orig = (batch["x"].permute(0, 2, 3, 1) + 1.0) * 127.5
    orig = orig.detach().cpu().numpy().astype(np.uint8)
    viz_images = [orig]

    def postprocess(x: Tensor):
        x = (x.permute(0, 2, 3, 1) + 1.0) * 127.5  # channels last, [0,255]
        return x.detach().cpu().numpy()

    # Generate samples using flow matching
    rf = RectifiedFlow()
    
    # Prepare parents dictionary for flow matching
    if args.concat_pa and "pa" in batch:
        # For concatenated format - would need splitting logic
        parents = None
    else:
        # For separate format
        parents = {
            'digit': batch.get('digit'),
            'thickness': batch.get('thickness'), 
            'intensity': batch.get('intensity')
        }
    
    # Generate unconditional samples
    with torch.no_grad():
        # Create a copy of args for unconditional sampling
        uncond_args = copy.deepcopy(args)
        uncond_args.num_samples = min(bs, 8)
        uncond_args.cfg_scale = 0.0  # Unconditional
        
        uncond_samples = model.sample(
            rf=rf,
            ars=uncond_args,
            parents=None  # Unconditional
        )
        uncond_viz = postprocess(uncond_samples)
        viz_images.append(uncond_viz.astype(np.uint8))
        
        # Generate conditional samples if we have parents
        if parents is not None and any(v is not None for v in parents.values()):
            # Create a copy of args for conditional sampling
            cond_args = copy.deepcopy(args)
            cond_args.num_samples = min(bs, 8)
            
            # Slice the parent tensors to match num_samples
            cond_parents = {}
            for key, value in parents.items():
                if value is not None:
                    cond_parents[key] = value[:min(bs, 8)]
                else:
                    cond_parents[key] = None
            
            cond_samples = model.sample(
                rf=rf,
                ars=cond_args,
                parents=cond_parents
            )
            cond_viz = postprocess(cond_samples)
            viz_images.append(cond_viz.astype(np.uint8))

    # Save visualization to tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        # This assumes there's a writer available globally or passed in
        # For now, we'll just return the images without saving
        pass
    except ImportError:
        pass
    except ImportError:
        pass
    
    # Save visualization to disk
    try:
        import imageio
        import os
        
        # Ensure all image arrays have the same shape for visualization
        if len(viz_images) > 1:
            # Get the target shape from the original images (first array)
            target_shape = viz_images[0].shape
            
            # Resize or pad all images to match target shape
            for j, img in enumerate(viz_images):
                if img.shape != target_shape:
                    # If dimensions are different, we need to resize
                    if img.shape[1:3] != target_shape[1:3]:  # H, W different
                        # For now, let's crop or pad to match
                        curr_h, curr_w = img.shape[1], img.shape[2]
                        target_h, target_w = target_shape[1], target_shape[2]
                        
                        if curr_h < target_h or curr_w < target_w:
                            # Pad if smaller
                            pad_h = max(0, target_h - curr_h)
                            pad_w = max(0, target_w - curr_w)
                            pad_top = pad_h // 2
                            pad_bottom = pad_h - pad_top
                            pad_left = pad_w // 2
                            pad_right = pad_w - pad_left
                            
                            viz_images[j] = np.pad(img, 
                                                 ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                                 mode='constant', constant_values=0)
                        elif curr_h > target_h or curr_w > target_w:
                            # Crop if larger
                            crop_h = min(curr_h, target_h)
                            crop_w = min(curr_w, target_w)
                            start_h = (curr_h - crop_h) // 2
                            start_w = (curr_w - crop_w) // 2
                            
                            viz_images[j] = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
        
        # Ensure all image arrays have the same batch size for visualization
        max_bs = max(img.shape[0] for img in viz_images)
        for j, img in enumerate(viz_images):
            s = img.shape[0]
            if s < max_bs:
                pad = np.zeros((max_bs - s, *img.shape[1:])).astype(np.uint8)
                viz_images[j] = np.concatenate([img, pad], axis=0)
        
        # Get final dimensions
        final_shape = viz_images[0].shape
        n_rows = len(viz_images)
        max_bs, h, w, c = final_shape
        
        # Concatenate all images and save to disk
        im = (
            np.concatenate(viz_images, axis=0)
            .reshape((n_rows, max_bs, h, w, c))
            .transpose([0, 2, 1, 3, 4])
            .reshape([n_rows * h, max_bs * w, c])
        )
        imageio.imwrite(os.path.join(args.save_dir, f"flow_viz-{args.iter}.png"), im)
    except ImportError:
        pass
    
    return viz_images