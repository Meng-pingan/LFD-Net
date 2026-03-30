import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FractionalDerivativeConv(nn.Module):
    """可学习分数阶微分卷积模块 - 按波段共享α参数

    使用unfold + 矩阵乘法替代Python双重循环
    速度提升约10-50倍，同时保持数值精度
    """

    def __init__(self, num_bands, kernel_size=7, alpha_init=0.5):
        """
        Args:
            num_bands: 波段数
            kernel_size: 卷积核大小
            alpha_init: α的初始值 (0, 2)，默认0.5（半阶微分）
                       - 较小的值(0.1-0.3): 更温和的微分，适合高质量数据
                       - 中等的值(0.4-0.6): 标准半阶微分
                       - 较大的值(0.7-1.0): 更强的微分，适合噪声数据
        """
        super().__init__()
        self.num_bands = num_bands
        self.K = kernel_size

        # 将alpha_init转换为alpha_raw的初始值
        # alpha = 2.0 * sigmoid(alpha_raw)，所以 alpha_raw = logit(alpha/2.0)
        # logit(x) = log(x / (1-x))
        alpha_ratio = alpha_init / 2.0  # 归一化到(0, 1)
        alpha_ratio = max(0.01, min(0.99, alpha_ratio))  # 避免数值问题
        alpha_raw_init = np.log(alpha_ratio / (1 - alpha_ratio))

        # 每个波段一个可学习的α参数
        self.alpha_raw = nn.Parameter(torch.ones(num_bands) * alpha_raw_init)

        # 可学习的能量缩放因子，初始化为10以补偿分数阶微分的幅度损失
        self.scale = nn.Parameter(torch.ones(num_bands) * 10.0)

    def _generate_all_kernels(self, alpha):
        """批量生成所有波段的卷积核 [C, K] - 向量化实现"""
        k = torch.arange(self.K, device=alpha.device, dtype=alpha.dtype)  # [K]

        # 扩展维度进行广播: alpha [C, 1], k [1, K]
        alpha_exp = alpha.unsqueeze(1)  # [C, 1]
        k_exp = k.unsqueeze(0)  # [1, K]

        # 计算有效mask (alpha - k + 1 > 0)
        valid_mask = (alpha_exp - k_exp + 1) > 0  # [C, K]

        # 在计算lgamma之前，先把无效位置替换成安全值，避免lgamma产生NaN
        alpha_k_diff = alpha_exp - k_exp + 1  # [C, K]
        alpha_k_diff_safe = torch.where(valid_mask, alpha_k_diff, torch.ones_like(alpha_k_diff))

        # 计算log系数 (使用广播)
        log_coef = (torch.lgamma(alpha_exp + 1)
                   - torch.lgamma(k_exp + 1)
                   - torch.lgamma(alpha_k_diff_safe))  # [C, K]

        # 计算权重
        sign = ((-1.0) ** k_exp)  # [1, K]
        weights = sign * torch.exp(log_coef)  # [C, K]

        # 应用有效mask，无效位置设为0
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        # 归一化保持能量
        weights_sum = torch.abs(weights).sum(dim=1, keepdim=True)  # [C, 1]
        weights_normalized = weights / (weights_sum + 1e-8)

        # 对于全零的行（weights_sum < 1e-8），设置中心为1
        zero_rows = (weights_sum.squeeze() < 1e-8)
        if zero_rows.any():
            center_idx = self.K // 2
            # 创建单位脉冲
            identity_kernel = torch.zeros(self.K, device=alpha.device, dtype=alpha.dtype)
            identity_kernel[center_idx] = 1.0
            weights_normalized[zero_rows] = identity_kernel

        return weights_normalized  # [C, K]

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 约束alpha到(0, 2)范围
        alpha = 2.0 * torch.sigmoid(self.alpha_raw)  # [C]

        # 批量生成所有卷积核
        kernels = self._generate_all_kernels(alpha)  # [C, K]

        # 重塑: [B, C, H, W] -> [B*H*W, C]
        x_reshape = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # 使用unfold + 矩阵乘法替代双重循环
        half_k = self.K // 2

        # 对光谱维度进行padding（复制边界值）
        # x_reshape: [B*H*W, C] -> [B*H*W, 1, C] -> pad -> [B*H*W, 1, C+K-1]
        x_padded = F.pad(x_reshape.unsqueeze(1), (half_k, half_k), mode='replicate')
        x_padded = x_padded.squeeze(1)  # [B*H*W, C+K-1]

        # 使用unfold提取滑动窗口: [B*H*W, C, K]
        x_unfolded = x_padded.unfold(1, self.K, 1)  # [B*H*W, C, K]

        # 与卷积核相乘并求和（向量化操作）
        # kernels: [C, K] -> [1, C, K]
        output = (x_unfolded * kernels.unsqueeze(0)).sum(dim=2)  # [B*H*W, C]

        # 检查NaN
        if torch.isnan(output).any():
            print("Warning: NaN detected in FractionalDerivativeConv output, using input instead")
            output = x_reshape

        # 重塑回原始形状
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 应用可学习的能量缩放，恢复分数阶微分丢失的幅度信息
        output = output * self.scale.view(1, -1, 1, 1)

        return output

    def get_alpha(self):
        """获取约束后的alpha值用于可视化"""
        with torch.no_grad():
            return 2.0 * torch.sigmoid(self.alpha_raw)

    def get_scale(self):
        """获取能量缩放因子用于可视化"""
        with torch.no_grad():
            return self.scale


class SpatialAttentionModule(nn.Module):
    """空间注意力模块 
    学习空间权重图，用于边缘保持和细节增强
    """

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
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # 输出: [B, 1, H, W] 空间权重图
        return self.attention_net(x)

class MediumSpectralBranch(nn.Module):
    """中等复杂度光谱分支：2层1x1卷积"""

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
    """中等复杂度空间分支：2层3x3卷积"""

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
    """中等复杂度光谱-空间双分支编码器

    设计理念：
    1. 光谱分支(1x1卷积)：2层学习波段间关系
    2. 空间分支(3x3卷积)：2层学习空间上下文
    3. 温度Softmax + 最小权重保护：防止分支权重不平衡
    """

    def __init__(self, num_bands, num_endmembers, hidden_channels=32, hidden_dim=32,
                 dropout=0.1, temperature=2.0, min_weight=0.2):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.temperature = temperature
        self.min_weight = min_weight

        # 光谱分支：2层1x1卷积
        self.spectral_branch = MediumSpectralBranch(
            num_bands, hidden_channels, hidden_dim, dropout
        )

        # 空间分支：2层3x3卷积
        self.spatial_branch = MediumSpatialBranch(
            num_bands, hidden_channels, hidden_dim, dropout
        )

        # 分支注意力：学习两个分支的重要性权重（不含Softmax）
        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, 2),
        )

        # 融合层：直接从拼接特征到输出
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, num_endmembers, 1),
        )

    def forward(self, x):
        # x: [B, C, H, W]

        # 双分支特征提取
        f_spectral = self.spectral_branch(x)  # [B, hidden_dim, H, W]
        f_spatial = self.spatial_branch(x)    # [B, hidden_dim, H, W]

        # 拼接特征
        f_concat = torch.cat([f_spectral, f_spatial], dim=1)  # [B, hidden_dim*2, H, W]

        # 计算分支注意力logits
        logits = self.branch_attention(f_concat)  # [B, 2]

        soft_weights = F.softmax(logits / self.temperature, dim=1)

        remaining = 1.0 - 2 * self.min_weight
        branch_weights = self.min_weight + remaining * soft_weights

        # # 提取各个分支的权重，并重塑为 [B, 1, 1, 1] 形状，以便在 H 和 W 维度上广播
        # # 使用 -1 自动推导 Batch 维度，比写死 B 更稳健
        # w_spec = branch_weights[:, 0].view(-1, 1, 1, 1)
        # w_spat = branch_weights[:, 1].view(-1, 1, 1, 1)

        # # 将动态计算的权重乘到对应的特征图上
        # f_spectral_weighted = f_spectral * w_spec
        # f_spatial_weighted = f_spatial * w_spat

        # # 拼接加权后的特征
        # f_weighted_concat = torch.cat([f_spectral_weighted, f_spatial_weighted], dim=1)

        # out = self.fusion(f_weighted_concat)  # [B, P, H, W]
        # # 融合输出
        out = self.fusion(f_concat)  # [B, P, H, W]

        return out, branch_weights

    def get_branch_weights(self, x):
        """获取分支权重用于可视化"""
        with torch.no_grad():
            f_spectral = self.spectral_branch(x)
            f_spatial = self.spatial_branch(x)
            f_concat = torch.cat([f_spectral, f_spatial], dim=1)
            logits = self.branch_attention(f_concat)
            soft_weights = F.softmax(logits / self.temperature, dim=1)
            remaining = 1.0 - 2 * self.min_weight
            return self.min_weight + remaining * soft_weights


class LFDNet(nn.Module):
    """LFD-Manifold Net - 结合分数阶微分、空间注意力和双分支编码器的高级模型
    """

    def __init__(self, num_bands, num_endmembers, dropout=0.1, kernel_size=7,
                 hidden_channels=32, hidden_dim=32, alpha_init=0.5):
        """
        Args:
            alpha_init: 分数阶微分的α初始值，默认0.5
                       可根据数据集特性调整：
                       - 高质量数据(如Moffett): 0.2-0.3 (温和微分)
                       - 标准数据(如Samson): 0.4-0.6 (标准微分)
                       - 复杂数据(如Houston): 0.6-0.8 (强微分)
        """
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers

        # 分数阶微分增强模块（继承自V3）
        self.frac_conv = FractionalDerivativeConv(num_bands, kernel_size, alpha_init=alpha_init)

        # 空间注意力模块（继承自V3）
        self.spatial_attention = SpatialAttentionModule(num_bands)

        # 中等复杂度光谱-空间双分支编码器
        self.encoder = SpectralSpatialEncoderV2(
            num_bands, num_endmembers, hidden_channels, hidden_dim, dropout,
            temperature=2.0, min_weight=0.2
        )

        # Softmax保证ASC约束
        self.softmax = nn.Softmax(dim=1)

        # 线性解码器 - 权重即为端元矩阵
        self.decoder = nn.Conv2d(num_endmembers, num_bands, 1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]

        # 强制解码器权重非负 (ANC约束)
        with torch.no_grad():
            self.decoder.weight.data.clamp_(min=0.0)

        # 1. 分数阶微分增强
        x_enhanced = self.frac_conv(x)

        # 2. 空间注意力
        spatial_weights = self.spatial_attention(x)  # [B, 1, H, W]

        # 3. 门控融合
        x_fused = x_enhanced * (1 - spatial_weights) + x * spatial_weights

        # 4. 中等复杂度光谱-空间编码
        abundance_logits, encoder_branch_weights = self.encoder(x_fused)

        # 5. Softmax得到丰度图 (ASC约束)
        abundance = self.softmax(abundance_logits)

        # 6. 线性解码重构
        reconstruction = self.decoder(abundance)

        return abundance, reconstruction, spatial_weights, encoder_branch_weights

    def get_endmembers(self):
        """提取端元矩阵 [C, P]"""
        with torch.no_grad():
            E = self.decoder.weight.data.squeeze()
            return E

    def get_alpha(self):
        """获取分数阶微分的alpha值"""
        return self.frac_conv.get_alpha()

    def get_scale(self):
        """获取能量缩放因子"""
        return self.frac_conv.get_scale()

    def get_spatial_attention(self, x):
        """获取空间注意力图"""
        with torch.no_grad():
            return self.spatial_attention(x)

    def get_encoder_branch_weights(self, x):
        """获取编码器分支权重"""
        with torch.no_grad():
            x_enhanced = self.frac_conv(x)
            spatial_weights = self.spatial_attention(x)
            x_fused = x_enhanced * (1 - spatial_weights) + x * spatial_weights
            return self.encoder.get_branch_weights(x_fused)

