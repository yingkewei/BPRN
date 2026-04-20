'''
RDP-SACB Hybrid Model - Optimized Encoder with SACB at Intermediate Scales
Integrates SACB spatial-adaptive convolution blocks into RDP network.
SACB is applied ONLY at intermediate scales (1/4 and 1/8 resolution) as feature refinement.

Key Design:
- Layer0 (full res): Standard conv + Edge enhancement (NO SACB)
- Layer1 (1/2 res): Dilated conv + Edge guidance (NO SACB)
- Layer2 (1/4 res): Dilated conv + SimplifiedSACB refinement
- Layer3 (1/8 res): Dilated conv + LightweightSACB refinement

Author: AI Assistant
Date: 2025-12-19
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
import math
from einops import rearrange, reduce


# ==================== 基础模块（原封保留/小改） ====================

class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer"""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrates a vector field via scaling and squaring."""

    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ConvInsBlock(nn.Module):
    """Convolutional block with InstanceNorm and LeakyReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.main = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ResBlock(nn.Module):
    """VoxRes module"""

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


# ==================== 改进模块 ====================

class ASPP3DLite(nn.Module):
    """Lightweight 3D ASPP with dilations (1,2,3)"""

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 3)):
        super().__init__()
        branch_channels = max(out_channels // (len(dilations) + 1), 1)
        self.b1 = nn.Conv3d(in_channels, branch_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv3d(in_channels, branch_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.bn = nn.InstanceNorm3d(branch_channels * (len(dilations) + 1))
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv3d(branch_channels * (len(dilations) + 1), out_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.InstanceNorm3d(out_channels)
        self.proj_act = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = [self.b1(x)] + [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.act(self.bn(x))
        x = self.proj(x)
        x = self.proj_act(self.proj_bn(x))
        return x


class EnhancedConvBlock(nn.Module):
    """增强的卷积块，使用标准卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class BoundaryEnhancement(nn.Module):
    """边界增强模块"""

    def __init__(self, channels):
        super().__init__()
        self.edge_conv_x = nn.Conv3d(channels, max(channels // 4, 1), 3, 1, 1)
        self.edge_conv_y = nn.Conv3d(channels, max(channels // 4, 1), 3, 1, 1)
        self.edge_conv_z = nn.Conv3d(channels, max(channels // 4, 1), 3, 1, 1)
        self.edge_conv_xyz = nn.Conv3d(channels, max(channels // 4, 1), 3, 1, 1)
        self.fusion_conv = nn.Conv3d(max(channels // 4, 1) * 4, channels, 1)

    def forward(self, x):
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)
        edge_z = self.edge_conv_z(x)
        edge_xyz = self.edge_conv_xyz(x)
        edge_features = torch.cat([edge_x, edge_y, edge_z, edge_xyz], dim=1)
        enhanced = self.fusion_conv(edge_features)
        return x + enhanced


# ==================== SACB 模块（保留原来的两个变体） ====================

class SimplifiedSACB(nn.Module):
    """
    简化版SACB模块 - 不依赖外部K-means库
    """

    def __init__(self, in_channels, out_channels, num_clusters=4, kernel_size=3, residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_clusters = num_clusters
        self.kernel_size = kernel_size
        self.residual = residual

        # 可学习的聚类中心（在特征空间）
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_channels))

        # 输入投影（增强特征）
        self.proj_in = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(0.1)
        )

        # 基础卷积权重（所有聚类共享）
        self.base_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.base_weight, mode='fan_out', nonlinearity='leaky_relu')

        # 为每个聚类生成权重调制因子的MLP
        hidden_dim = 128
        self.weight_modulator = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, kernel_size ** 3),
            nn.Sigmoid()
        )

        # 为每个聚类生成偏置的MLP
        self.bias_generator = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_channels)
        )

        # 输出归一化和激活
        self.norm_out = nn.InstanceNorm3d(out_channels)
        self.act_out = nn.LeakyReLU(0.1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        identity = x

        # 1. 输入投影
        x = self.proj_in(x)

        # 2. 计算每个空间位置的特征向量（用于聚类分配）
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # 3. 计算特征到聚类中心的相似度（软分配）
        cluster_sim = torch.matmul(x_flat, self.cluster_centers.t())
        cluster_weights = F.softmax(cluster_sim / np.sqrt(C), dim=-1)

        # 4. 为每个聚类生成自适应卷积核
        out = torch.zeros(B, self.out_channels, D, H, W, device=x.device, dtype=x.dtype)

        for k in range(self.num_clusters):
            center_k = self.cluster_centers[k:k + 1]
            weight_mod = self.weight_modulator(center_k)
            weight_mod = weight_mod.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
            modulated_weight = self.base_weight * weight_mod
            bias_k = self.bias_generator(center_k).squeeze(0)
            conv_out = F.conv3d(x, modulated_weight, bias=bias_k, padding=self.kernel_size // 2)
            cluster_weight_k = cluster_weights[:, :, k:k + 1]
            cluster_weight_k = cluster_weight_k.view(B, 1, D, H, W)
            out = out + conv_out * cluster_weight_k

        out = self.norm_out(out)
        out = self.act_out(out)

        if self.residual and self.in_channels == self.out_channels:
            out = out + identity

        return out


class LightweightSACB(nn.Module):
    """
    轻量级SACB - 使用分组卷积减少计算量
    """

    def __init__(self, in_channels, out_channels, num_clusters=3, kernel_size=3, residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_clusters = num_clusters
        self.residual = residual

        # 通道分组（减少参数量）
        self.groups = min(4, in_channels)

        # 可学习的聚类中心
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_channels // self.groups))

        # 输入投影
        self.proj_in = ConvInsBlock(in_channels, in_channels, 3, 1, 1)

        # 分组卷积（减少参数）
        self.group_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    padding=kernel_size // 2, groups=self.groups, bias=False)

        # 聚类特定的1x1卷积（调制）
        self.cluster_modulators = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, 1, bias=True)
            for _ in range(num_clusters)
        ])

        self.norm_out = nn.InstanceNorm3d(out_channels)
        self.act_out = nn.LeakyReLU(0.1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        identity = x

        # 投影
        x = self.proj_in(x)

        # 分组卷积
        base_out = self.group_conv(x)

        # 计算聚类分配（使用全局平均池化的特征）
        x_pooled = F.adaptive_avg_pool3d(x, 1).view(B, C)

        # 分组特征
        x_pooled_grouped = x_pooled.view(B, self.groups, C // self.groups)
        x_pooled_grouped = x_pooled_grouped.mean(dim=1)

        # 计算到聚类中心的距离
        cluster_sim = torch.matmul(x_pooled_grouped, self.cluster_centers.t())
        cluster_weights = F.softmax(cluster_sim, dim=-1)

        # 加权融合不同聚类的调制
        out = torch.zeros_like(base_out)
        for k in range(self.num_clusters):
            modulated = self.cluster_modulators[k](base_out)
            weight_k = cluster_weights[:, k:k + 1, None, None, None]
            out = out + modulated * weight_k

        out = self.norm_out(out)
        out = self.act_out(out)

        if self.residual and self.in_channels == self.out_channels:
            out = out + identity

        return out


# ==================== 三路并行编码器（ParallelTripleEncoder） ====================

class ParallelTripleEncoder(nn.Module):
    """
    Optimized Encoder with SACB only at intermediate scales (1/4 and 1/8 resolution)
    - Layer0 (full res): Standard conv + Edge enhancement (NO SACB)
    - Layer1 (1/2 res): Dilated conv + Edge guidance (NO SACB)
    - Layer2 (1/4 res): Dilated conv + SimplifiedSACB refinement + Edge guidance
    - Layer3 (1/8 res): Dilated conv + LightweightSACB refinement + Edge attention
    
    SACB acts as feature refinement blocks that preserve spatial resolution and channel size.
    """

    def __init__(self, in_channel=1, first_out_channel=16, use_lightweight=True):
        super().__init__()
        c = first_out_channel

        # Shared initial projection
        self.initial_proj = ConvInsBlock(in_channel, c, 3, 1, 1)

        # ---- Edge branch (shallow extraction + downsample for guidance) ----
        self.edge_shallow = nn.Sequential(
            ConvInsBlock(c, c, 3, 1, 1),
            BoundaryEnhancement(c)
        )
        # Downsample projections for edge guidance propagation
        self.edge_down1 = nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1, bias=False)
        self.edge_down2 = nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1, bias=False)
        self.edge_down3 = nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1, bias=False)
        # 1x1 projections for edge injection at each layer
        self.edge_proj0 = nn.Conv3d(c, c // 4 if c // 4 > 0 else 1, 1, bias=False)
        self.edge_proj1 = nn.Conv3d(c, c // 2 if c // 2 > 0 else 1, 1, bias=False)
        self.edge_proj2 = nn.Conv3d(c, c // 2 if c // 2 > 0 else 1, 1, bias=False)

        # ---- Layer0 (full res) - NO SACB ----
        self.conv0 = nn.Sequential(
            ConvInsBlock(c, c, 3, 1, 1),
            ConvInsBlock(c, c, 3, 1, 1)
        )
        edge_ch0 = c // 4 if c // 4 > 0 else 1
        self.fuse0 = nn.Sequential(
            nn.Conv3d(c + edge_ch0, c, 1, bias=False),
            nn.InstanceNorm3d(c),
            nn.ReLU(inplace=True)
        )

        # ---- Layer1 (1/2 res) - NO SACB ----
        self.down1 = nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1, bias=False)
        self.dilated1_convs = nn.ModuleList([
            nn.Conv3d(2 * c, 2 * c // 2, kernel_size=3, padding=d, dilation=d, bias=False) 
            for d in (1, 2)
        ])
        edge_ch1 = c // 2 if c // 2 > 0 else 1
        self.fuse1 = nn.Sequential(
            nn.Conv3d(2 * c // 2 + edge_ch1, 2 * c, 1, bias=False),
            nn.InstanceNorm3d(2 * c),
            nn.ReLU(inplace=True)
        )

        # ---- Layer2 (1/4 res) - SimplifiedSACB as refinement ----
        self.down2 = nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1, bias=False)
        self.dilated2_convs = nn.ModuleList([
            nn.Conv3d(4 * c, 4 * c // 3, kernel_size=3, padding=d, dilation=d, bias=False) 
            for d in (1, 2, 3)
        ])
        # SimplifiedSACB for feature refinement (preserves 4*c channels)
        self.sacb2_refine = SimplifiedSACB(4 * c, 4 * c, num_clusters=4, residual=True)
        edge_ch2 = c // 2 if c // 2 > 0 else 1
        self.fuse2 = nn.Sequential(
            nn.Conv3d(4 * c // 3 + edge_ch2, 4 * c, 1, bias=False),
            nn.InstanceNorm3d(4 * c),
            nn.ReLU(inplace=True)
        )

        # ---- Layer3 (1/8 res) - LightweightSACB as refinement ----
        self.down3 = nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1, bias=False)
        self.dilated3_convs = nn.ModuleList([
            nn.Conv3d(8 * c, 8 * c // 3, kernel_size=3, padding=d, dilation=d, bias=False) 
            for d in (1, 2, 3)
        ])
        # LightweightSACB for feature refinement (preserves 8*c channels)
        self.sacb3_refine = LightweightSACB(8 * c, 8 * c, num_clusters=4, residual=True)
        # Edge-guided attention
        self.edge_attn_conv = nn.Sequential(
            nn.Conv3d(c, 8 * c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(8 * c),
            nn.Sigmoid()
        )
        self.fuse3 = nn.Sequential(
            nn.Conv3d(8 * c // 3, 8 * c, 1, bias=False),
            nn.InstanceNorm3d(8 * c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.initial_proj(x)

        # ----- Edge branch (shallow extraction) -----
        edge0 = self.edge_shallow(x0)  # [B, c, D, H, W]
        edge1 = self.edge_down1(edge0)  # [B, c, D/2, H/2, W/2]
        edge2 = self.edge_down2(edge1)  # [B, c, D/4, H/4, W/4]
        edge3 = self.edge_down3(edge2)  # [B, c, D/8, H/8, W/8]

        # ===== Layer0 (full res) - NO SACB =====
        feat0 = self.conv0(x0)
        e0p = self.edge_proj0(edge0)
        e0p = F.interpolate(e0p, size=feat0.shape[2:], mode='trilinear', align_corners=True)
        concat0 = torch.cat([feat0, e0p], dim=1)
        out0 = self.fuse0(concat0)

        # ===== Layer1 (1/2 res) - NO SACB =====
        x1 = self.down1(out0)
        dil1_parts = [conv(x1) for conv in self.dilated1_convs]
        dil1 = sum(dil1_parts)
        e1p = self.edge_proj1(edge1)
        e1p = F.interpolate(e1p, size=dil1.shape[2:], mode='trilinear', align_corners=True)
        concat1 = torch.cat([dil1, e1p], dim=1)
        out1 = self.fuse1(concat1)

        # ===== Layer2 (1/4 res) - SimplifiedSACB refinement =====
        x2 = self.down2(out1)
        dil2_parts = [conv(x2) for conv in self.dilated2_convs]
        dil2 = sum(dil2_parts)
        e2p = self.edge_proj2(edge2)
        e2p = F.interpolate(e2p, size=dil2.shape[2:], mode='trilinear', align_corners=True)
        concat2 = torch.cat([dil2, e2p], dim=1)
        feat2 = self.fuse2(concat2)
        # SACB refinement (preserves spatial resolution and channels)
        out2 = self.sacb2_refine(feat2)

        # ===== Layer3 (1/8 res) - LightweightSACB refinement =====
        x3 = self.down3(out2)
        dil3_parts = [conv(x3) for conv in self.dilated3_convs]
        dil3 = sum(dil3_parts)
        feat3 = self.fuse3(dil3)
        # SACB refinement (preserves spatial resolution and channels)
        feat3_refined = self.sacb3_refine(feat3)
        # Edge-guided attention
        edge3_aligned = F.interpolate(edge3, size=feat3_refined.shape[2:], mode='trilinear', align_corners=True)
        attn = self.edge_attn_conv(edge3_aligned)
        out3 = feat3_refined * attn

        return [out0, out1, out2, out3]


# ==================== SACB 自适应融合（不变） ====================
class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()

        c = channel

        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )

    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class SACB_AdaptiveFusion(nn.Module):
    """
    使用SACB的自适应特征融合
    """

    def __init__(self, in_channels, use_lightweight=True):
        super().__init__()
        SACB_Block = LightweightSACB if use_lightweight else SimplifiedSACB
        self.reduce = nn.Conv3d(in_channels, in_channels, 1)
        self.sacb = SACB_Block(in_channels, in_channels, num_clusters=3, residual=False)
        self.final_conv = ConvInsBlock(in_channels, in_channels, 3, 1)

    def forward(self, fixed_fm, warped_fm, decoder_fm):
        concat_fm = torch.cat([fixed_fm, warped_fm, decoder_fm], dim=1)
        x = self.reduce(concat_fm)
        x = self.sacb(x)
        x = self.final_conv(x)
        return x


class BEARefine(nn.Module):
    """
    Boundary-Error-Aware velocity refinement
    
    Refines velocity field based on:
    1. Registration error (fixed_feat - warped_feat)
    2. Boundary information
    3. Gating mechanism for adaptive correction
    
    Used after velocity prediction and before VecInt integration
    """
    
    def __init__(self, feat_channels, alpha=0.01):
        super().__init__()
        
        # Project registration error to 4 channels
        self.error_proj = nn.Conv3d(feat_channels, 4, kernel_size=1)
        
        # Project boundary features to 4 channels
        self.edge_proj = nn.Conv3d(feat_channels, 4, kernel_size=1)
        
        # Refinement logic: predict velocity correction
        self.refine_logic = nn.Sequential(
            nn.Conv3d(3 + 4 + 4, 16, kernel_size=3, padding=1),  # v + error + edge
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 3, kernel_size=1)
        )
        
        # Gate: modulate correction based on boundary strength
        self.gate_conv = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Learnable scaling factor (initialized conservatively)
        self.gamma = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, v, fixed_feat, warped_feat, boundary_feat):
        """
        Args:
            v: velocity field [B, 3, D, H, W]
            fixed_feat: fixed image features [B, C, D, H, W]
            warped_feat: warped moving features [B, C, D, H, W]
            boundary_feat: boundary features [B, C, D, H, W]
        
        Returns:
            refined velocity field [B, 3, D, H, W]
        """
        # Compute registration error
        error = fixed_feat - warped_feat
        proj_error = self.error_proj(error)
        
        # Extract boundary information
        proj_edge = self.edge_proj(boundary_feat)
        
        # Boundary-based gating
        gate = self.gate_conv(proj_edge)  # [B, 1, D, H, W]
        
        # Predict velocity correction
        concat = torch.cat([v, proj_error, proj_edge], dim=1)
        delta_v = self.refine_logic(concat)
        
        # Apply correction: v + gamma * (gate * delta_v)
        # - gate: high at boundaries, low in smooth regions
        # - gamma: learnable scaling (starts at 0.01)
        return v + self.gamma * (gate * delta_v)


# ==================== RDP-SACB混合模型（主模型，使用 ParallelTripleEncoder） ====================

class BPRN(nn.Module):
    def __init__(self, inshape=(160, 192, 160), flow_multiplier=1., in_channel=1, channels=16,
                 use_lightweight_sacb=True, use_bea_refine=False, bea_alpha=0.01):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape
        self.use_bea_refine = use_bea_refine
        c = self.channels

        # Use optimized encoder with SACB only at intermediate scales (1/4 and 1/8)
        self.encoder_moving = ParallelTripleEncoder(in_channel=in_channel,
                                                    first_out_channel=c,
                                                    use_lightweight=use_lightweight_sacb)
        self.encoder_fixed = ParallelTripleEncoder(in_channel=in_channel,
                                                   first_out_channel=c,
                                                   use_lightweight=use_lightweight_sacb)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # Warp和Diff模块
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        # 瓶颈层：ASPP + SACB
        SACB_Block = LightweightSACB if use_lightweight_sacb else SimplifiedSACB
        self.cconv_4 = nn.Sequential(
            ASPP3DLite(16 * c, 8 * c, dilations=(1, 2, 3)),
            SACB_Block(8 * c, 8 * c, num_clusters=4, residual=True)
        )

        # 变形场预测层（第4层）
        self.defconv4 = nn.Conv3d(8 * c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))

        self.dconv4 = nn.Sequential(
            SACB_Block(3 * 8 * c, 8 * c, num_clusters=4, residual=False),
            EnhancedConvBlock(8 * c, 8 * c)
        )

        # 第3层解码器 (2次递归 - 第1次用SACB，第2次用标准卷积)
        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.adaptive_fusion_3 = SACB_AdaptiveFusion(3 * 4 * c, use_lightweight=use_lightweight_sacb)
        self.adaptive_fusion_3_recur = nn.Sequential(
            nn.Conv3d(3 * 4 * c, 3 * 4 * c, 1),
            ConvInsBlock(3 * 4 * c, 3 * 4 * c, 3, 1)
        )
        self.defconv3 = nn.Conv3d(3 * 4 * c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = SACB_Block(3 * 4 * c, 4 * c, num_clusters=4, residual=False)

        # 第2层解码器 (3次递归 - 第一次用SACB，后续用标准卷积)
        self.upconv2 = UpConvBlock(3 * 4 * c, 2 * c, 4, 2)
        # 第一次融合：使用SACB
        self.adaptive_fusion_2_first = SACB_AdaptiveFusion(3 * 2 * c, use_lightweight=use_lightweight_sacb)
        # 递归融合：使用标准卷积（轻量高效）
        self.adaptive_fusion_2_recur = nn.Sequential(
            nn.Conv3d(3 * 2 * c, 3 * 2 * c, 1),
            ConvInsBlock(3 * 2 * c, 3 * 2 * c, 3, 1)
        )
        self.defconv2 = nn.Conv3d(3 * 2 * c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        # 第一次dconv：使用SACB
        self.dconv2_first = SACB_Block(3 * 2 * c, 2 * c, num_clusters=3, residual=False)
        # 递归dconv：使用标准卷积
        self.dconv2_recur = nn.Sequential(
            ConvInsBlock(3 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(2 * c, 2 * c, 3, 1)
        )

        # 第1层解码器 (无递归 - 使用标准卷积即可)
        self.upconv1 = UpConvBlock(3 * 2 * c, c, 4, 2)
        self.adaptive_fusion_1 = nn.Sequential(
            nn.Conv3d(3 * c, 3 * c, 1),
            ConvInsBlock(3 * c, 3 * c, 3, 1)
        )
        self.defconv1 = nn.Conv3d(3 * c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))
        
        # BEARefine: Boundary-Error-Aware velocity refinement at full resolution
        if self.use_bea_refine:
            self.bea_refine = BEARefine(feat_channels=c, alpha=bea_alpha)

    def forward(self, moving, fixed):
        # Encoding stage - SACB applied only at 1/4 and 1/8 resolution inside encoder
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)

        # ========== Decoding stage - Maintains RDP's recursive refinement framework ==========
        # 第4层（最粗糙层）
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)
        w = self.diff[3](v)

        # 第3层 - 递归优化 (2次递归：第1次用SACB，第2次用标准卷积)
        D3 = self.upconv3(C4)
        flow = self.upsample_trilin(2 * (self.warp[3](flow, w) + w))
        
        # 第1次递归 - 使用SACB
        warped = self.warp[2](M3, flow)
        C3 = self.adaptive_fusion_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w) + w
        
        # 第2次递归 - 使用标准卷积
        warped = self.warp[2](M3, flow)
        D3 = self.dconv3(C3)
        C3 = self.adaptive_fusion_3_recur(torch.cat([F3, warped, D3], dim=1))
        v = self.defconv3(C3)
        w = self.diff[2](v)

        # 第2层 - 递归优化 (3次递归：第1次用SACB，后2次用标准卷积)
        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2 * (self.warp[2](flow, w) + w))
        
        # 第1次递归 - 使用SACB
        warped = self.warp[1](M2, flow)
        C2 = self.adaptive_fusion_2_first(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 第2次递归 - 使用标准卷积
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_first(C2)
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 第3次递归 - 使用标准卷积
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_recur(C2)  # C2 is 3*2*c channels
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)

        # 第1层 - 最终输出 (无递归)
        D1 = self.upconv1(C2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.adaptive_fusion_1(torch.cat([F1, warped, D1], dim=1))
        v = self.defconv1(C1)
        
        # BEARefine: Refine velocity based on error and boundary
        if self.use_bea_refine:
            v = self.bea_refine(
                v,           # velocity field
                F1,          # fixed features (boundary info)
                warped,      # warped moving features
                F1           # use fixed features as boundary features
            )
        
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w

        # 最终warp
        y_moved = self.warp[0](moving, flow)
        return y_moved, flow


# ==================== 轻量级变体（也使用并行编码器，A选项） ====================

class BPRN_Lite(nn.Module):
    """
    Lightweight RDP-SACB Hybrid Model
    - Reduced channel count
    - SACB only at intermediate scales (1/4 and 1/8 resolution)
    - Maintains decoder logic
    """

    def __init__(self, inshape=(160, 192, 160), flow_multiplier=1., in_channel=1, channels=12):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.step = 7
        self.inshape = inshape
        c = self.channels

        # Use optimized encoder with SACB only at intermediate scales
        self.encoder_moving = ParallelTripleEncoder(in_channel=in_channel, first_out_channel=c, use_lightweight=True)
        self.encoder_fixed = ParallelTripleEncoder(in_channel=in_channel, first_out_channel=c, use_lightweight=True)

        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2 ** i for s in inshape]))
            self.diff.append(VecInt([s // 2 ** i for s in inshape]))

        # 瓶颈层：简化ASPP + 轻量SACB
        self.cconv_4 = nn.Sequential(
            ASPP3DLite(16 * c, 8 * c, dilations=(1, 2)),
            LightweightSACB(8 * c, 8 * c, num_clusters=2, residual=True)
        )

        # 变形场预测（同标准版）
        self.defconv4 = nn.Conv3d(8 * c, 3, 3, 1, 1)
        self.defconv4.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv4.weight.shape))
        self.defconv4.bias = nn.Parameter(torch.zeros(self.defconv4.bias.shape))
        self.dconv4 = nn.Sequential(
            LightweightSACB(3 * 8 * c, 8 * c, num_clusters=2, residual=False),
            EnhancedConvBlock(8 * c, 8 * c)
        )

        # Layer3 decoder (2 recursions - 1st uses SACB, 2nd uses standard conv)
        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)
        self.adaptive_fusion_3 = SACB_AdaptiveFusion(3 * 4 * c, use_lightweight=True)
        self.adaptive_fusion_3_recur = nn.Sequential(
            nn.Conv3d(3 * 4 * c, 3 * 4 * c, 1),
            ConvInsBlock(3 * 4 * c, 3 * 4 * c, 3, 1)
        )
        self.defconv3 = nn.Conv3d(3 * 4 * c, 3, 3, 1, 1)
        self.defconv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv3.weight.shape))
        self.defconv3.bias = nn.Parameter(torch.zeros(self.defconv3.bias.shape))
        self.dconv3 = LightweightSACB(3 * 4 * c, 4 * c, num_clusters=2, residual=False)

        # Layer2 decoder (3 recursions - 1st uses SACB, rest use standard conv)
        self.upconv2 = UpConvBlock(3 * 4 * c, 2 * c, 4, 2)
        self.adaptive_fusion_2_first = SACB_AdaptiveFusion(3 * 2 * c, use_lightweight=True)
        self.adaptive_fusion_2_recur = nn.Sequential(
            nn.Conv3d(3 * 2 * c, 3 * 2 * c, 1),
            ConvInsBlock(3 * 2 * c, 3 * 2 * c, 3, 1)
        )
        self.defconv2 = nn.Conv3d(3 * 2 * c, 3, 3, 1, 1)
        self.defconv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv2.weight.shape))
        self.defconv2.bias = nn.Parameter(torch.zeros(self.defconv2.bias.shape))
        self.dconv2_first = LightweightSACB(3 * 2 * c, 2 * c, num_clusters=2, residual=False)
        self.dconv2_recur = nn.Sequential(
            ConvInsBlock(3 * 2 * c, 2 * c, 3, 1),
            ConvInsBlock(2 * c, 2 * c, 3, 1)
        )

        # Layer1 decoder (no recursion - standard conv)
        self.upconv1 = UpConvBlock(3 * 2 * c, c, 4, 2)
        self.adaptive_fusion_1 = nn.Sequential(
            nn.Conv3d(3 * c, 3 * c, 1),
            ConvInsBlock(3 * c, 3 * c, 3, 1)
        )
        self.defconv1 = nn.Conv3d(3 * c, 3, 3, 1, 1)
        self.defconv1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.defconv1.weight.shape))
        self.defconv1.bias = nn.Parameter(torch.zeros(self.defconv1.bias.shape))

    def forward(self, moving, fixed):
        # Encoding - SACB applied only at 1/4 and 1/8 resolution inside encoder
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)

        # Decoding (maintains recursive refinement logic)
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow = self.defconv4(C4)
        flow = self.diff[3](flow)
        warped = self.warp[3](M4, flow)
        C4 = self.dconv4(torch.cat([F4, warped, C4], dim=1))
        v = self.defconv4(C4)
        w = self.diff[3](v)

        # Layer3 - Recursive refinement (2 recursions: 1st SACB, 2nd standard)
        D3 = self.upconv3(C4)
        flow = self.upsample_trilin(2 * (self.warp[3](flow, w) + w))
        
        # 1st recursion - SACB
        warped = self.warp[2](M3, flow)
        C3 = self.adaptive_fusion_3(F3, warped, D3)
        v = self.defconv3(C3)
        w = self.diff[2](v)
        flow = self.warp[2](flow, w) + w
        
        # 2nd recursion - standard conv
        warped = self.warp[2](M3, flow)
        D3 = self.dconv3(C3)
        C3 = self.adaptive_fusion_3_recur(torch.cat([F3, warped, D3], dim=1))
        v = self.defconv3(C3)
        w = self.diff[2](v)

        # Layer2 - Recursive refinement (3 recursions: 1st SACB, rest standard)
        D2 = self.upconv2(C3)
        flow = self.upsample_trilin(2 * (self.warp[2](flow, w) + w))
        
        # 1st recursion - SACB
        warped = self.warp[1](M2, flow)
        C2 = self.adaptive_fusion_2_first(F2, warped, D2)
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 2nd recursion - standard conv
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_first(C2)
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)
        flow = self.warp[1](flow, w) + w
        
        # 3rd recursion - standard conv
        warped = self.warp[1](M2, flow)
        D2 = self.dconv2_recur(C2)  # C2 is 3*2*c channels
        C2 = self.adaptive_fusion_2_recur(torch.cat([F2, warped, D2], dim=1))
        v = self.defconv2(C2)
        w = self.diff[1](v)

        # Layer1 - Final output (no recursion)
        D1 = self.upconv1(C2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, w) + w))
        warped = self.warp[0](M1, flow)
        C1 = self.adaptive_fusion_1(torch.cat([F1, warped, D1], dim=1))
        v = self.defconv1(C1)
        w = self.diff[0](v)
        flow = self.warp[0](flow, w) + w

        y_moved = self.warp[0](moving, flow)
        return y_moved, flow


# ==================== 测试块（保持） ====================
if __name__ == '__main__':
    print("=" * 80)
    print("RDP-SACB Hybrid Model Test (Parallel Triple Encoder - A+C)")
    print("=" * 80)

    # 测试SimplifiedSACB
    sacb = SimplifiedSACB(in_channels=32, out_channels=32, num_clusters=4)
    x = torch.randn(1, 32, 20, 24, 20)
    out = sacb(x)
    print(f"   SimplifiedSACB Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    # 测试LightweightSACB
    lite_sacb = LightweightSACB(in_channels=32, out_channels=32, num_clusters=3)
    out = lite_sacb(x)
    print(f"   LightweightSACB Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    # 测试完整模型
    model = BPRN(inshape=(64, 80, 64), channels=8, use_lightweight_sacb=True)
    moving = torch.randn(1, 1, 64, 80, 64)
    fixed = torch.randn(1, 1, 64, 80, 64)
    print(f"   Input shape: {moving.shape}")
    with torch.no_grad():
        warped, flow = model(moving, fixed)
    print(f"   Warped shape: {warped.shape}")
    print(f"   Flow shape: {flow.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试轻量级模型
    lite_model = BPRN_Lite(inshape=(64, 80, 64), channels=6)
    with torch.no_grad():
        warped, flow = lite_model(moving, fixed)
    print(f"   Lite Warped shape: {warped.shape}")
    print(f"   Lite Flow shape: {flow.shape}")
    print(f"   Lite Total parameters: {sum(p.numel() for p in lite_model.parameters()):,}")

    print("\n" + " = "*80)
    print("All tests done (use small inputs for sanity).")
    print("=" * 80)
