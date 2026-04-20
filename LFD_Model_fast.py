import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FractionalDerivativeConv(nn.Module):
    """可学习分数阶微分卷积模块"""
    def __init__(self, num_bands, kernel_size=7, alpha_init=0.5):
        super().__init__()
        self.num_bands = num_bands
        self.K = kernel_size

        alpha_ratio = alpha_init / 2.0  
        alpha_ratio = max(0.01, min(0.99, alpha_ratio))  
        alpha_raw_init = np.log(alpha_ratio / (1 - alpha_ratio))

        self.alpha_raw = nn.Parameter(torch.ones(num_bands) * alpha_raw_init)
        self.scale = nn.Parameter(torch.ones(num_bands) * 10.0)

    def _generate_all_kernels(self, alpha):
        k = torch.arange(self.K, device=alpha.device, dtype=alpha.dtype)  

        alpha_exp = alpha.unsqueeze(1)  
        k_exp = k.unsqueeze(0) 

        valid_mask = (alpha_exp - k_exp + 1) > 0  

        alpha_k_diff = alpha_exp - k_exp + 1  
        alpha_k_diff_safe = torch.where(valid_mask, alpha_k_diff, torch.ones_like(alpha_k_diff))

        log_coef = (torch.lgamma(alpha_exp + 1)
                   - torch.lgamma(k_exp + 1)
                   - torch.lgamma(alpha_k_diff_safe))  

        # 计算权重
        sign = ((-1.0) ** k_exp)  
        weights = sign * torch.exp(log_coef)  

        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        weights_sum = torch.abs(weights).sum(dim=1, keepdim=True)  
        weights_normalized = weights / (weights_sum + 1e-8)

        zero_rows = (weights_sum.squeeze() < 1e-8)
        if zero_rows.any():
            center_idx = self.K // 2
            identity_kernel = torch.zeros(self.K, device=alpha.device, dtype=alpha.dtype)
            identity_kernel[center_idx] = 1.0
            weights_normalized[zero_rows] = identity_kernel

        return weights_normalized  

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        alpha = 2.0 * torch.sigmoid(self.alpha_raw)  # [C]
        kernels = self._generate_all_kernels(alpha)  # [C, K]

        x_reshape = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # 使用unfold + 矩阵乘法
        half_k = self.K // 2

        # x_reshape: [B*H*W, C] -> [B*H*W, 1, C] -> pad -> [B*H*W, 1, C+K-1]
        x_padded = F.pad(x_reshape.unsqueeze(1), (half_k, half_k), mode='replicate')
        x_padded = x_padded.squeeze(1)  # [B*H*W, C+K-1]

        x_unfolded = x_padded.unfold(1, self.K, 1)  # [B*H*W, C, K]

        # kernels: [C, K] -> [1, C, K]
        output = (x_unfolded * kernels.unsqueeze(0)).sum(dim=2)  

        if torch.isnan(output).any():
            print("Warning: NaN detected in FractionalDerivativeConv output, using input instead")
            output = x_reshape

        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        output = output * self.scale.view(1, -1, 1, 1)

        return output

    def get_alpha(self):
        with torch.no_grad():
            return 2.0 * torch.sigmoid(self.alpha_raw)

    def get_scale(self):
        with torch.no_grad():
            return self.scale


class SpatialAttentionModule(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.attention_net(x)

class MediumSpectralBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.branch(x)


class MediumSpatialBranch(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(hidden_channels),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.branch(x)


class SpectralSpatialEncoderV2(nn.Module):
    def __init__(self, num_bands, num_endmembers, hidden_channels=32, hidden_dim=32,
                 dropout=0.1, temperature=2.0, min_weight=0.2):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.temperature = temperature
        self.min_weight = min_weight

        self.spectral_branch = MediumSpectralBranch(
            num_bands, hidden_channels, hidden_dim, dropout
        )

        self.spatial_branch = MediumSpatialBranch(
            num_bands, hidden_channels, hidden_dim, dropout
        )

        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, 2),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, num_endmembers, 1),
        )

    def forward(self, x):
        # x: [B, C, H, W]

        f_spectral = self.spectral_branch(x)  # [B, hidden_dim, H, W]
        f_spatial = self.spatial_branch(x)    # [B, hidden_dim, H, W]

        f_concat = torch.cat([f_spectral, f_spatial], dim=1)  

        logits = self.branch_attention(f_concat)  # [B, 2]
        soft_weights = F.softmax(logits / self.temperature, dim=1)

        remaining = 1.0 - 2 * self.min_weight
        branch_weights = self.min_weight + remaining * soft_weights

        out = self.fusion(f_concat)  # [B, P, H, W]

        return out, branch_weights

    def get_branch_weights(self, x):
        with torch.no_grad():
            f_spectral = self.spectral_branch(x)
            f_spatial = self.spatial_branch(x)
            f_concat = torch.cat([f_spectral, f_spatial], dim=1)
            logits = self.branch_attention(f_concat)
            soft_weights = F.softmax(logits / self.temperature, dim=1)
            remaining = 1.0 - 2 * self.min_weight
            return self.min_weight + remaining * soft_weights


class LFDNet(nn.Module):
    def __init__(self, num_bands, num_endmembers, dropout=0.1, kernel_size=7,
                 hidden_channels=32, hidden_dim=32, alpha_init=0.5):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers

        self.frac_conv = FractionalDerivativeConv(num_bands, kernel_size, alpha_init=alpha_init)

        self.spatial_attention = SpatialAttentionModule(num_bands)

        self.encoder = SpectralSpatialEncoderV2(
            num_bands, num_endmembers, hidden_channels, hidden_dim, dropout,
            temperature=2.0, min_weight=0.2
        )

        self.softmax = nn.Softmax(dim=1)

        # 线性解码器
        self.decoder = nn.Conv2d(num_endmembers, num_bands, 1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        with torch.no_grad():
            self.decoder.weight.data.clamp_(min=0.0)

        x_enhanced = self.frac_conv(x)

        spatial_weights = self.spatial_attention(x)  # [B, 1, H, W]
        x_fused = x_enhanced * (1 - spatial_weights) + x * spatial_weights

        abundance_logits, encoder_branch_weights = self.encoder(x_fused)
        abundance = self.softmax(abundance_logits)

        reconstruction = self.decoder(abundance)

        return abundance, reconstruction, spatial_weights, encoder_branch_weights

    def get_endmembers(self):
        with torch.no_grad():
            E = self.decoder.weight.data.squeeze()
            return E

    def get_alpha(self):
        return self.frac_conv.get_alpha()

    def get_scale(self):
        return self.frac_conv.get_scale()

    def get_spatial_attention(self, x):
        with torch.no_grad():
            return self.spatial_attention(x)

    def get_encoder_branch_weights(self, x):
        with torch.no_grad():
            x_enhanced = self.frac_conv(x)
            spatial_weights = self.spatial_attention(x)
            x_fused = x_enhanced * (1 - spatial_weights) + x * spatial_weights
            return self.encoder.get_branch_weights(x_fused)

