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
    """MiniUnet的下采样层 Resnet
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 线性层，用于时间编码换通道 [B, dim] -> [B, in_channels]
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        # 降采样
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2)

        self.in_channels = in_channels

    def forward(self, x, temb):
        # x: [B, C, H, W]
        res = x
        x += self.fc(temb)[:, :, None, None]  # [B, in_channels, 1, 1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
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
    """MiniUnet的上采样层
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16,
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 线性层，用于时间编码换通道
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, temb):
        # 上采样
        if self.upsample:
            x = self.upsample(x)
        res = x

        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x


class MiddleLayer(nn.Module):
    """MiniUnet的中间层
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

        # 线性层，用于时间编码换通道
        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x

        x += self.fc(temb)[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        if self.shortcut is not None:
            x = self.shortcut(x)
        x = x + res

        return x


class DeeperUnet(nn.Module):
    """Fixed Deeper UNet for MNIST with thickness conditioning
    """

    def __init__(self, args: Hparams):
        super(DeeperUnet, self).__init__()
        base_channels=args.base_channels
        time_emb_dim = args.time_emb_dim 
        if time_emb_dim is None:
            time_emb_dim = base_channels

        self.time_emb_dim = time_emb_dim
        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # Down blocks - Fixed architecture
        # Level 1: 28x28 -> 14x14
        self.down1 = nn.ModuleList([
            DownLayer(base_channels,
                      base_channels * 2,
                      time_emb_dim=time_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 2,
                      base_channels * 2,
                      time_emb_dim=time_emb_dim)
        ])
        self.maxpool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14

        # Level 2: 14x14 -> 7x7
        self.down2 = nn.ModuleList([
            DownLayer(base_channels * 2,
                      base_channels * 4,
                      time_emb_dim=time_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 4,
                      base_channels * 4,
                      time_emb_dim=time_emb_dim)
        ])
        self.maxpool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7

        # Level 3: 7x7 -> 3x3 (using stride 2 for odd dimensions)
        self.down3 = nn.ModuleList([
            DownLayer(base_channels * 4,
                      base_channels * 8,
                      time_emb_dim=time_emb_dim,
                      downsample=False),
            DownLayer(base_channels * 8,
                      base_channels * 8,
                      time_emb_dim=time_emb_dim)
        ])
        # Use conv with stride 2 - 7x7 -> 3x3 (more natural for odd dimensions)
        self.maxpool3 = nn.Conv2d(base_channels * 8, base_channels * 8, 
                                  kernel_size=3, stride=2, padding=0)  # 7x7 -> 3x3

        # Middle layer
        self.middle = MiddleLayer(base_channels * 8,
                                  base_channels * 8,
                                  time_emb_dim=time_emb_dim)

        # Up blocks - Fixed with proper dimensions
        # Level 1: 3x3 -> 7x7 (to match x3)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 
                                           kernel_size=3, stride=2, padding=0, output_padding=0)  # 3x3 -> 7x7
        
        self.up1 = nn.ModuleList([
            UpLayer(
                base_channels * 16,  # concat from down3: 8*base + 8*base
                base_channels * 4,
                time_emb_dim=time_emb_dim,
                upsample=False),
            UpLayer(base_channels * 4,
                    base_channels * 4,
                    time_emb_dim=time_emb_dim)
        ])
        
        # Level 2: 7x7 -> 14x14 (to match x2)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 
                                           kernel_size=2, stride=2)  # 7x7 -> 14x14
        self.up2 = nn.ModuleList([
            UpLayer(base_channels * 8,  # concat from down2: 4*base + 4*base
                    base_channels * 2,
                    time_emb_dim=time_emb_dim,
                    upsample=False),
            UpLayer(base_channels * 2,
                    base_channels * 2,
                    time_emb_dim=time_emb_dim)
        ])
        
        # Level 3: 14x14 -> 28x28 (to match x1)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 
                                           kernel_size=2, stride=2)  # 14x14 -> 28x28
        self.up3 = nn.ModuleList([
            UpLayer(base_channels * 4,  # concat from down1: 2*base + 2*base
                    base_channels,
                    time_emb_dim=time_emb_dim,
                    upsample=False),
            UpLayer(base_channels,
                    base_channels,
                    time_emb_dim=time_emb_dim)
        ])

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=1, padding=0)

    def time_emb(self, t, dim):
        """对时间进行正弦函数的编码"""
        # 把t映射到[0, 1000]
        t = t * 1000
        freqs = torch.pow(10000, -torch.linspace(0, 1, dim // 2)).to(t.device)
        args = t[:, None] * freqs[None, :]  # Proper broadcasting
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def label_emb(self, y, dim):
        """对类别标签进行编码"""
        y = y * 1000
        freqs = torch.pow(10000, -torch.linspace(0, 1, dim // 2)).to(y.device)
        args = y[:, None] * freqs[None, :]  # Proper broadcasting
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)

        return torch.cat([sin_emb, cos_emb], dim=-1)

    def thickness_emb(self, thickness, dim):
        """对thickness进行编码 - Balanced scaling"""
        # Normalize thickness to reasonable range first
        thickness_norm = torch.clamp(thickness, 0.1, 5.0)  # Clamp to reasonable range
        thickness_scaled = thickness_norm  # Remove the *200 scaling - keep raw normalized values

        freqs = torch.pow(6, -torch.linspace(0, 1, dim // 2)).to(thickness.device)
        args = thickness_scaled[:, None] * freqs[None, :]  # Proper broadcasting
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def intensity_emb(self, intensity, dim):
        """对intensity进行编码 - Balanced scaling"""
        # Normalize intensity to reasonable range first
        intensity_norm = torch.clamp(intensity, 50.0, 250.0)  # Updated for mean=150, std=50
        intensity_scaled = (intensity_norm - 49.0) / 200.0 * 5.0  # Scale to [0, 5] to match thickness range

        freqs = torch.pow(6, -torch.linspace(0, 1, dim // 2)).to(intensity.device)
        args = intensity_scaled[:, None] * freqs[None, :]  # Proper broadcasting
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def forward(self, x, t, parents=None):
        """前向传播函数 - Fixed embedding handling"""
        # x:(B, C, H, W) - Input should be 28x28
        x = self.conv_in(x)  # 28x28 -> 28x28 with base_channels
        
        # Create separate embeddings and concatenate them
        temb = self.time_emb(t, self.base_channels)
        
        # Handle parents being None (unconditional case)
        if parents is not None:
            y = parents.get('digit')
            thickness = parents.get('thickness')
            intensity = parents.get('intensity')
        else:
            y = None
            thickness = None
            intensity = None

        # Handle label conditioning
        if y is not None:
            yemb = self.label_emb(y, self.base_channels)
            yemb[y == -1] = 0.0
            temb = temb + yemb  
        
        # Handle thickness conditioning
        if thickness is not None:
            thickness_emb = self.thickness_emb(thickness, self.base_channels)
            thickness_emb[thickness == -1] = 0.0
            temb = temb + thickness_emb  
        
        # Handle intensity conditioning
        if intensity is not None:
            intensity_emb = self.intensity_emb(intensity, self.base_channels)
            intensity_emb[intensity == -1] = 0.0
            temb = temb + intensity_emb  
        
        # Down path with skip connections
        for layer in self.down1:
            x = layer(x, temb)
        x1 = x  # Skip connection 1: [B, 64, 28, 28]
        x = self.maxpool1(x)  # [B, 64, 14, 14]
        
        for layer in self.down2:
            x = layer(x, temb)
        x2 = x  # Skip connection 2: [B, 128, 14, 14]
        x = self.maxpool2(x)  # [B, 128, 7, 7]
        
        for layer in self.down3:
            x = layer(x, temb)
        x3 = x  # Skip connection 3: [B, 256, 7, 7]
        x = self.maxpool3(x)  # [B, 256, 3, 3]

        # Middle layer
        x = self.middle(x, temb)  # [B, 256, 3, 3]

        # Up path with skip connections
        x = self.upsample1(x)  # [B, 256, 7, 7] - Now matches x3 exactly
        x = torch.cat([x, x3], dim=1)  # [B, 512, 7, 7]
        for layer in self.up1:
            x = layer(x, temb)  # [B, 128, 7, 7]
            
        x = self.upsample2(x)  # [B, 128, 14, 14]
        x = torch.cat([x, x2], dim=1)  # [B, 256, 14, 14]
        for layer in self.up2:
            x = layer(x, temb)  # [B, 64, 14, 14]
            
        x = self.upsample3(x)  # [B, 64, 28, 28]
        x = torch.cat([x, x1], dim=1)  # [B, 128, 28, 28]
        for layer in self.up3:
            x = layer(x, temb)  # [B, 32, 28, 28]

        x = self.conv_out(x)  # [B, 1, 28, 28]
        return x
    
    def sample(self, rf, ars: Hparams, parents=None):
        """Generate samples using the trained flow matching model with thickness conditioning

        Args:
            rf: RectifiedFlow instance
            ars: Hyperparameters containing num_samples, cfg_scale, num_steps, device
            parents: Dictionary with 'digit', 'thickness', 'intensity' keys
        """
        num_samples = ars.num_samples
        cfg_scale = ars.cfg_scale
        num_steps = ars.num_steps
        device = ars.device
        
        # Extract individual components from parents if provided
        y = parents.get('digit') if parents is not None else None
        thickness = parents.get('thickness') if parents is not None else None
        intensity = parents.get('intensity') if parents is not None else None
        
        if y is not None:
            assert len(y.shape) == 1, 'y must be 1D tensor'
            if y.shape[0] == 1:
                y = y.repeat(num_samples).reshape(num_samples)
            y = y.to(device)

        if thickness is not None:
            assert len(thickness.shape) == 1, 'thickness must be 1D tensor'
            if thickness.shape[0] == 1:
                thickness = thickness.repeat(num_samples).reshape(num_samples)
            thickness = thickness.to(device)

        if intensity is not None:
            assert len(intensity.shape) == 1, 'intensity must be 1D tensor'
            if intensity.shape[0] == 1:
                intensity = intensity.repeat(num_samples).reshape(num_samples)
            intensity = intensity.to(device)
        
        # Reconstruct parents dictionary for the model
        sample_parents = {
            'digit': y,
            'thickness': thickness,
            'intensity': intensity
        } if any(x is not None for x in [y, thickness, intensity]) else None
        
        # Move model to device and set to eval mode
        model = self.to(device)
        model.eval()
        
        with torch.no_grad():
            dt = 1.0 / num_steps
            # Start from noise
            x_t = torch.randn(num_samples, 1, 28, 28, device=device)

            # Integrate the flow from t=0 to t=1
            for i in range(num_steps):
                t = torch.full((num_samples,), i * dt, device=device)
                v_pred_uncon = model(x_t, t, None)  # Unconditional
                v_pred_con = model(x_t, t, sample_parents)  # Conditional
                v_pred = v_pred_uncon + cfg_scale * (v_pred_con - v_pred_uncon)
                x_t = rf.euler(x_t, v_pred, dt)

            return x_t

        
