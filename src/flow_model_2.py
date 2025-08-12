import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from hps import Hparams

class RectifiedFlow:
  def euler(self,x_t,v,dt):
    x_t=x_t+v*dt
    return x_t
  def create_flow(self,x_1,t):
    x_0=torch.randn_like(x_1)
    t=t[:,None,None,None]
    x_t=t*x_1+(1-t)*x_0
    return x_t,x_0
  def mse_loss(self,x_1,x_0,v):
    loss=F.mse_loss(x_1-x_0,v)
    return loss


# Deeper UNet for MNIST 28*28 with thickness conditioning
class DownLayer(nn.Module):
    """Improved downsampling layer with better conditioning integration"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 cond_emb_dim=16,
                 downsample=False):
        super(DownLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)  
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()  

        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels)
        
        self.film_scale = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)
        self.film_shift = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        # 降采样
        self.downsample = downsample
        if downsample:
            self.pool = nn.AvgPool2d(2)  # AvgPool instead of MaxPool

        self.in_channels = in_channels

    def forward(self, x, temb, cemb=None):
        # x: [B, C, H, W]
        res = x
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Apply conditioning via FiLM (Feature-wise Linear Modulation)
        if cemb is not None:
            combined_emb = torch.cat([temb, cemb], dim=-1)
        else:
            combined_emb = temb
            
        scale = self.film_scale(combined_emb)[:, :, None, None]
        shift = self.film_shift(combined_emb)[:, :, None, None]
        x = x * (1 + scale) + shift
        
        x = self.act(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)

        x = x + res

        if self.downsample:
            x = self.pool(x)

        return x


class UpLayer(nn.Module):
    """Improved upsampling layer with better conditioning"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
                 cond_emb_dim=16,
                 upsample=False):
        super(UpLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels)
        
        # FiLM layers
        self.film_scale = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)
        self.film_shift = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, temb, cemb=None):
        # 上采样
        if self.upsample:
            x = self.upsample_layer(x)
        res = x

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Apply conditioning via FiLM
        if cemb is not None:
            combined_emb = torch.cat([temb, cemb], dim=-1)
        else:
            combined_emb = temb
            
        scale = self.film_scale(combined_emb)[:, :, None, None]
        shift = self.film_shift(combined_emb)[:, :, None, None]
        x = x * (1 + scale) + shift
        
        x = self.act(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x


class MiddleLayer(nn.Module):
    """Improved middle layer with better conditioning"""

    def __init__(self, in_channels, out_channels, time_emb_dim=16, cond_emb_dim=16):
        super(MiddleLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels)
        
        # FiLM layers
        self.film_scale = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)
        self.film_shift = nn.Linear(time_emb_dim + cond_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb, cemb=None):
        res = x

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Apply conditioning via FiLM
        if cemb is not None:
            combined_emb = torch.cat([temb, cemb], dim=-1)
        else:
            combined_emb = temb
            
        scale = self.film_scale(combined_emb)[:, :, None, None]
        shift = self.film_shift(combined_emb)[:, :, None, None]
        x = x * (1 + scale) + shift
        
        x = self.act(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x


class FlowUnet(nn.Module):
    """Improved UNet with better conditioning for 32x32 input"""

    def __init__(self, args: Hparams):
        super(FlowUnet, self).__init__()
        base_channels = args.base_channels
        time_emb_dim = args.time_emb_dim 
        if time_emb_dim is None:
            time_emb_dim = base_channels

        self.time_emb_dim = time_emb_dim
        self.base_channels = base_channels
        self.cond_emb_dim = time_emb_dim  # Match conditioning embedding dimension to time embedding
        
        # Store expected input resolution and concatenation setting
        self.input_res = getattr(args, 'input_res', 32)
        self.concat_pa = getattr(args, 'concat_pa', False)

        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # Individual projection dimensions to ensure final combined size matches cond_emb_dim
        individual_emb_dim = self.cond_emb_dim // 3  # Divide among 3 components: digit, thickness, intensity
        
        # Conditioning for concatenated format [batch, 12]: 10 digit + 1 thickness + 1 intensity
        # This handles both concat_pa=True (single tensor) and concat_pa=False (dict format)
        self.digit_proj = nn.Sequential(
            nn.Linear(10, individual_emb_dim),  # Project one-hot digit to rich embedding
            nn.SiLU(),
            nn.Linear(individual_emb_dim, individual_emb_dim)
        )
        self.thickness_proj = nn.Sequential(
            nn.Linear(1, individual_emb_dim),
            nn.SiLU(),
            nn.Linear(individual_emb_dim, individual_emb_dim)
        )
        self.intensity_proj = nn.Sequential(
            nn.Linear(1, individual_emb_dim),
            nn.SiLU(),
            nn.Linear(individual_emb_dim, individual_emb_dim)
        )
        
        # Direct projection for concatenated conditioning [batch, 12] -> [batch, cond_emb_dim]
        self.concat_cond_proj = nn.Sequential(
            nn.Linear(12, self.cond_emb_dim),
            nn.SiLU(),
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim)
        )
        
        # Final conditioning projection (for dict format)
        self.cond_final_proj = nn.Sequential(
            nn.Linear(individual_emb_dim * 3, self.cond_emb_dim),  # 3 * individual_emb_dim -> cond_emb_dim
            nn.SiLU(),
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim)
        )

        # Down blocks - Updated with conditioning
        # Level 1: 32x32 -> 16x16
        self.down1 = nn.ModuleList([
            DownLayer(base_channels,
                      base_channels * 2,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 2,
                      base_channels * 2,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim)
        ])
        self.maxpool1 = nn.AvgPool2d(2)  # 32x32 -> 16x16

        # Level 2: 16x16 -> 8x8
        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2,
                      base_channels * 4,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 4,
                      base_channels * 4,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim)
        ])
        self.maxpool2 = nn.AvgPool2d(2)  # 16x16 -> 8x8

        # Level 3: 8x8 -> 4x4
        self.down3 = nn.ModuleList([
            DownLayer(base_channels * 4,
                      base_channels * 8,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 8,
                      base_channels * 8,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim)
        ])
        self.maxpool3 = nn.AvgPool2d(2)  # 8x8 -> 4x4

        # Level 4: Keep at 4x4 but increase channels (bottleneck)
        self.down4 = nn.ModuleList([
            DownLayer(base_channels * 8,
                      base_channels * 16,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 16,
                      base_channels * 16,
                      time_emb_dim=time_emb_dim,
                      cond_emb_dim=self.cond_emb_dim)
        ])

        # Middle layer - stay at 4x4 with maximum channels
        self.middle = MiddleLayer(base_channels * 16,
                                  base_channels * 32,
                                  time_emb_dim=time_emb_dim,
                                  cond_emb_dim=self.cond_emb_dim)

        # Up blocks - Updated with conditioning
        # Level 1: 4x4 -> 4x4 (reduce channels first)
        self.up1 = nn.ModuleList([
            UpLayer(
                base_channels * 48,  # concat: 32*base + 16*base
                base_channels * 8,
                time_emb_dim=time_emb_dim,
                cond_emb_dim=self.cond_emb_dim,
                upsample=False),
            UpLayer(base_channels * 8,
                    base_channels * 8,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim)
        ])
        
        # Level 2: 4x4 -> 8x8
        self.upsample2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 
                                           kernel_size=2, stride=2)  # 4x4 -> 8x8
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 16,  # concat: 8*base + 8*base
                    base_channels * 4,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim,
                    upsample=False),
            UpLayer(base_channels * 4,
                    base_channels * 4,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim)
        ])
        
        # Level 3: 8x8 -> 16x16
        self.upsample3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 
                                           kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.up3 = nn.ModuleList([
            UpLayer(base_channels * 8,  # concat: 4*base + 4*base
                    base_channels * 2,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim,
                    upsample=False),
            UpLayer(base_channels * 2,
                    base_channels * 2,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim)
        ])
        
        # Level 4: 16x16 -> 32x32
        self.upsample4 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 
                                           kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.up4 = nn.ModuleList([
            UpLayer(base_channels * 4,  # concat: 2*base + 2*base
                    base_channels,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim,
                    upsample=False),
            UpLayer(base_channels,
                    base_channels,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=self.cond_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        """Improved time embedding with better frequency coverage"""
        # Scale t to [0, 1000] range for better frequency representation
        t = t * 1000
        half_dim = dim // 2
        freqs = torch.exp(-torch.linspace(0, np.log(10000), half_dim, device=t.device))
        args = t[:, None] * freqs[None, :]
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def create_conditioning_embedding(self, parents, batch_size=None):
        """Create conditioning embedding supporting concatenated format:
        
        Format: parents is a tensor [batch, 12]
          - parents[:, 0:1]: thickness 
          - parents[:, 1:2]: intensity
          - parents[:, 2:12]: digit one-hot (10 dims)
        
        Returns: (batch, cond_emb_dim) combined embedding
        """
        device = next(self.parameters()).device
        
        if parents is None:
            if batch_size is None:
                batch_size = 1
            return torch.zeros(batch_size, self.cond_emb_dim, device=device)
        
        # Format: Concatenated tensor [batch, 12] (concat_pa=True)
        if self.concat_pa:
            if not isinstance(parents, torch.Tensor):
                raise ValueError(f"When concat_pa=True, expected parents to be torch.Tensor, got {type(parents)}")
            if parents.dim() != 2 or parents.shape[1] != 12:
                raise ValueError(f"When concat_pa=True, expected parents shape [batch, 12], got {parents.shape}")

            actual_batch_size = parents.shape[0]
            
            # Handle batch size mismatch by repeating if necessary
            if batch_size is not None and actual_batch_size != batch_size:
                if actual_batch_size == 1 and batch_size > 1:
                    parents = parents.repeat(batch_size, 1)
                    actual_batch_size = batch_size
                elif actual_batch_size != batch_size:
                    raise ValueError(f"Cannot handle batch size mismatch: parents has {actual_batch_size}, expected {batch_size}")
            
            parents_processed = parents.clone()
            
            # Handle thickness unconditional
            thickness_part = parents_processed[:, 0:1]  # [batch, 1] 
            thickness_unconditional_mask = (thickness_part.squeeze(1) == -1)  # [batch]
            thickness_part[thickness_unconditional_mask] = 0.0
            
            # Handle intensity unconditional  
            intensity_part = parents_processed[:, 1:2]  # [batch, 1]
            intensity_unconditional_mask = (intensity_part.squeeze(1) == -1)  # [batch]
            intensity_part[intensity_unconditional_mask] = 0.0
            
            # Handle digit unconditional: if all digit dims are -1, set to zeros
            digit_part = parents_processed[:, 2:12]  # [batch, 10]
            digit_unconditional_mask = (digit_part == -1).all(dim=1)  # [batch]
            digit_part[digit_unconditional_mask] = 0.0
            
            # Combine processed parts
            parents_clean = torch.cat([thickness_part, intensity_part, digit_part], dim=1)
            
            # Direct projection
            final_emb = self.concat_cond_proj(parents_clean)
            return final_emb
        else:
            raise ValueError("Conditioning format not supported. Set concat_pa=True to use concatenated tensor format.")
    
    def forward(self, x, t, parents=None):
        """Improved forward pass with better conditioning integration"""
        # Validate spatial size early
        h, w = x.shape[-2:]
        if (h, w) != (self.input_res, self.input_res):
            raise ValueError(f"FlowUnet expects input {(self.input_res, self.input_res)} but got {(h, w)}.")
        if h // 8 != 4:
            raise ValueError(f"Input size {h} not compatible with architecture depth (needs h/8 == 4 -> h == 32).")
        
        # Input convolution
        x = self.conv_in(x)  # 32x32 -> 32x32 with base_channels
        
        # Create time embedding
        temb = self.time_emb(t, self.time_emb_dim)
        
        # Create conditioning embedding with correct batch size
        batch_size = x.shape[0]
        cemb = self.create_conditioning_embedding(parents, batch_size)
        
        # Down path with skip connections - 4 levels
        # Level 1: 32x32 -> 16x16
        for layer in self.down1:
            x = layer(x, temb, cemb)
        x1 = x  # Skip connection 1: [B, 2*base, 32, 32]
        x = self.maxpool1(x)  # [B, 2*base, 16, 16]
        
        # Level 2: 16x16 -> 8x8
        for layer in self.down2:
            x = layer(x, temb, cemb)
        x2 = x  # Skip connection 2: [B, 4*base, 16, 16]
        x = self.maxpool2(x)  # [B, 4*base, 8, 8]
        
        # Level 3: 8x8 -> 4x4
        for layer in self.down3:
            x = layer(x, temb, cemb)
        x3 = x  # Skip connection 3: [B, 8*base, 8, 8]
        x = self.maxpool3(x)  # [B, 8*base, 4, 4]
        
        # Level 4: 4x4 (no more downsampling, just increase channels)
        for layer in self.down4:
            x = layer(x, temb, cemb)
        x4 = x  # Skip connection 4: [B, 16*base, 4, 4]

        # Middle layer - bottleneck at 4x4
        x = self.middle(x, temb, cemb)  # [B, 32*base, 4, 4]

        # Up path with skip connections - 4 levels
        # Level 1: 4x4 -> 4x4 (reduce channels)
        x = torch.cat([x, x4], dim=1)  # [B, 48*base, 4, 4]
        for layer in self.up1:
            x = layer(x, temb, cemb)  # [B, 8*base, 4, 4]
            
        # Level 2: 4x4 -> 8x8
        x = self.upsample2(x)  # [B, 8*base, 8, 8]
        x = torch.cat([x, x3], dim=1)  # [B, 16*base, 8, 8]
        for layer in self.up2:
            x = layer(x, temb, cemb)  # [B, 4*base, 8, 8]
            
        # Level 3: 8x8 -> 16x16
        x = self.upsample3(x)  # [B, 4*base, 16, 16]
        x = torch.cat([x, x2], dim=1)  # [B, 8*base, 16, 16]
        for layer in self.up3:
            x = layer(x, temb, cemb)  # [B, 2*base, 16, 16]
            
        # Level 4: 16x16 -> 32x32
        x = self.upsample4(x)  # [B, 2*base, 32, 32]
        x = torch.cat([x, x1], dim=1)  # [B, 4*base, 32, 32]
        for layer in self.up4:
            x = layer(x, temb, cemb)  # [B, base, 32, 32]

        x = self.conv_out(x)  # [B, 1, 32, 32]
        return x
    
    def sample(self, rf, ars: Hparams, parents=None):
        """Generate samples using the trained flow matching model with conditioning

        Args:
            rf: RectifiedFlow instance
            ars: Hyperparameters containing bs, num_steps, device
            parents: Tensor with shape [batch, 12] when concat_pa=True, or None for unconditional
        """
        bs = ars.bs
        num_steps = ars.num_steps
        device = ars.device
        
        # Check that concat_pa is True since model only supports concatenated format
        if not self.concat_pa:
            raise ValueError("Model requires concat_pa=True. Set this in your hyperparameters.")
        
        # Validate parents format
        if parents is not None:
            if not isinstance(parents, torch.Tensor):
                raise ValueError(f"When concat_pa=True, expected parents to be torch.Tensor, got {type(parents)}")
            if parents.dim() != 2 or parents.shape[1] != 12:
                raise ValueError(f"Expected parents shape [batch, 12], got {parents.shape}")
            
            # Ensure correct batch size
            if parents.shape[0] == 1 and bs > 1:
                parents = parents.repeat(bs, 1)
            elif parents.shape[0] != bs:
                raise ValueError(f"Parents batch size {parents.shape[0]} doesn't match expected batch size {bs}")
            
            sample_parents = parents.to(device)
        else:
            sample_parents = None
        
        # Move model to device and set to eval mode
        model = self.to(device)
        model.eval()
        
        with torch.no_grad():
            dt = 1.0 / num_steps
            x_t = torch.randn(bs, 1, self.input_res, self.input_res, device=device)
            for i in range(num_steps):
                t = torch.full((bs,), i * dt, device=device)
                
                # Simple conditional or unconditional sampling (no CFG)
                v_pred = model(x_t, t, sample_parents)
                x_t = rf.euler(x_t, v_pred, dt)

            return x_t
    def sample_with_cf(self, rf, x_0, args: Hparams, cf_parents=None):
        bs = x_0.shape[0]
        num_steps = args.num_steps
        device = args.device

        # Check that concat_pa is True since model only supports concatenated format
        if not self.concat_pa:
            raise ValueError("Model requires concat_pa=True. Set this in your hyperparameters.")

        # Validate cf_parents format
        if cf_parents is not None:
            if not isinstance(cf_parents, torch.Tensor):
                raise ValueError(f"When concat_pa=True, expected cf_parents to be torch.Tensor, got {type(cf_parents)}")
            if cf_parents.dim() != 2 or cf_parents.shape[1] != 12:
                raise ValueError(f"Expected cf_parents shape [batch, 12], got {cf_parents.shape}")
            
            # Ensure correct batch size
            if cf_parents.shape[0] == 1 and bs > 1:
                cf_parents = cf_parents.repeat(bs, 1)
            elif cf_parents.shape[0] != bs:
                raise ValueError(f"cf_parents batch size {cf_parents.shape[0]} doesn't match expected batch size {bs}")
            
            sample_parents = cf_parents.to(device)
        else:
            sample_parents = None
        
        # Move model to device and set to eval mode
        model = self.to(device)
        model.eval()
        with torch.no_grad():
            dt = 1.0 / num_steps
            x_t = x_0
            for i in range(num_steps):
                t = torch.full((bs,), i * dt, device=device)
                v_pred = model(x_t, t, sample_parents)
                x_t = rf.euler(x_t, v_pred, dt)
            return x_t
                
            
    def abduct(self, x, parents, args: Hparams, cf_parents=None):
        """
        Abduction step: Given observed x at t=1, infer the initial noise z at t=0
        using the learned flow. Optionally, allow counterfactual parents for the backward pass.

        Args:
            x: Observed data at t=1, shape (batch, 1, H, W)
            parents: Conditioning tensor [batch, 12] for the factual world
            args: Hparams object with num_steps, device, bs
            cf_parents: Conditioning tensor [batch, 12] for the counterfactual world (optional)

        Returns:
            z0: The inferred initial noise at t=0
        """
        num_steps = args.num_steps
        device = args.device
        bs = x.shape[0]
        
        # Check that concat_pa is True since model only supports concatenated format
        if not self.concat_pa:
            raise ValueError("Model requires concat_pa=True. Set this in your hyperparameters.")
        
        # Use counterfactual parents if provided, otherwise use factual parents
        target_parents = cf_parents if cf_parents is not None else parents
        
        # Validate parents format
        if target_parents is not None:
            if not isinstance(target_parents, torch.Tensor):
                raise ValueError(f"When concat_pa=True, expected parents to be torch.Tensor, got {type(target_parents)}")
            if target_parents.dim() != 2 or target_parents.shape[1] != 12:
                raise ValueError(f"Expected parents shape [batch, 12], got {target_parents.shape}")
            
            # Ensure correct batch size
            if target_parents.shape[0] != bs:
                raise ValueError(f"Parents batch size {target_parents.shape[0]} doesn't match input batch size {bs}")
            
            cond_parents = target_parents.to(device)
        else:
            cond_parents = None
        
        model = self.to(device)
        model.eval()
        dt = 1.0 / num_steps
        x_t = x.clone()
        
        with torch.no_grad():
            for i in reversed(range(num_steps)):
                t_prev = torch.full((bs,), i * dt, device=device)
                v_pred = model(x_t, t_prev, cond_parents)
                x_t = x_t - v_pred * dt
        return x_t






