import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sam2.build_sam import build_sam2
from sam2.utils.dca_utils import UpsampleConv
from sam2.utils.dca_utils import depthwise_projection
from sam2.utils.dca_utils import PoolEmbedding
from sam2.utils.dca_utils import ScaleDotProduct


# ==========================================
# 1. 跨尺度语义引导融合模块 (CSGF)
# ==========================================
class CSGF(nn.Module):
    def __init__(self, ch_in, ch_out, num_groups=2):
        super(CSGF, self).__init__()

        # 并联分组卷积
        self.group_conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True)
        self.group_conv2 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True)

        self.gelu = nn.GELU()
        self.group_norm1 = nn.GroupNorm(num_groups, ch_in)
        self.group_norm2 = nn.GroupNorm(num_groups, ch_in)

        # 逐点卷积 (Point-wise 1x1 Conv) 调整通道并深化特征
        self.conv1x1_1 = nn.Conv2d(ch_in * 2, ch_out * 4, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(ch_out * 4, ch_out, kernel_size=1)

        self.group_norm3 = nn.GroupNorm(num_groups, ch_out * 4)
        self.group_norm4 = nn.GroupNorm(num_groups, ch_out)

    def forward(self, x):
        x1 = self.gelu(self.group_norm1(self.group_conv1(x)))
        x2 = self.gelu(self.group_norm2(self.group_conv2(x)))

        # 通道拼接
        x = torch.cat([x1, x2], dim=1)

        # 逐点降维聚合
        x = self.gelu(self.group_norm3(self.conv1x1_1(x)))
        x = self.gelu(self.group_norm4(self.conv1x1_2(x)))
        return x


class Up(nn.Module):
    """上采样与 CSGF 引导融合层"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.csgf = CSGF(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.csgf(x)


# ==========================================
# 2. Hiera 骨干适配器 (Adapter)
# ==========================================
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        return self.block(promped)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel * 2, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        gap = x.view(b, c, -1).mean(dim=2)
        gmp = x.view(b, c, -1).max(dim=2)[0]
        y = torch.cat([gap, gmp], dim=1)
        y = self.sigmoid(self.fc2(self.fc1(y))).view(b, c, 1, 1)
        return x * y


# ==========================================
# 3. 多尺度感知增强模块 (MSPE)
# ==========================================
class MSPE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSPE, self).__init__()
        self.relu = nn.ReLU(True)
        self.se = SEBlock(out_channel)

        # 分支 0：1x1 卷积直接映射
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        # 分支 1：非对称卷积组合 (1x3, 3x1) + 空洞卷积 (d=3)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channel)
        )
        # 分支 2：非对称卷积组合 (1x5, 5x1) + 空洞卷积 (d=5)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channel)
        )
        # 分支 3：非对称卷积组合 (1x7, 7x1) + 空洞卷积 (d=7)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        x = self.se(x)
        return x


# ==========================================
# 4. 双重跨层级特征融合模块 (DCFF)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.projection = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        return self.projection(att)


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features, out_features=in_features, groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.projection = depthwise_projection(in_features=out_features, out_features=out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        return self.projection(att)


class DCFFBlock(nn.Module):
    def __init__(self, features, channel_head, spatial_head, spatial_att=True, channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att

        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.c_attention = nn.ModuleList([
                ChannelAttention(in_features=sum(features), out_features=feature, n_heads=head)
                for feature, head in zip(features, channel_head)
            ])

        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.s_attention = nn.ModuleList([
                SpatialAttention(in_features=sum(features), out_features=feature, n_heads=head)
                for feature, head in zip(features, spatial_head)
            ])

    def forward(self, x):
        x_ca = self.channel_attention(x) if self.channel_att else x
        x_sa = self.spatial_attention(x) if self.spatial_att else x
        return self.m_sum(x_ca, x_sa)

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        return self.m_apply(x_in, self.c_attention)

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        return self.m_apply(x_in, self.s_attention)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        if x[0].size()[2:] != y[0].size()[2:]:
            target_size = y[0].size()[2:]
            x = [F.interpolate(xi, size=target_size, mode='bilinear', align_corners=True) for xi in x]
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)


class DCFF(nn.Module):
    def __init__(self, features, strides, patch=28, channel_att=True, spatial_att=True, n=1,
                 channel_head=[1, 1, 1, 1], spatial_head=[4, 4, 4, 4]):
        super().__init__()
        self.patch = patch
        self.patch_avg = nn.ModuleList([
            PoolEmbedding(pooling=nn.AdaptiveAvgPool2d, patch=patch) for _ in features
        ])
        self.avg_map = nn.ModuleList([
            depthwise_projection(in_features=feature, out_features=feature, kernel_size=(1, 1), padding=(0, 0),
                                 groups=feature)
            for feature in features
        ])
        self.attention = nn.ModuleList([
            DCFFBlock(features=features, channel_head=channel_head, spatial_head=spatial_head,
                      channel_att=channel_att, spatial_att=spatial_att)
            for _ in range(n)
        ])
        self.upconvs = nn.ModuleList([
            UpsampleConv(in_features=feature, out_features=feature, kernel_size=(1, 1), padding=(0, 0),
                         norm_type=None, activation=False, scale=stride, conv='conv')
            for feature, stride in zip(features, strides)
        ])
        self.bn_relu = nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(feature), nn.ReLU()) for feature in features
        ])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        if x[0].size()[2:] != y[0].size()[2:]:
            target_size = y[0].size()[2:]
            x = [F.interpolate(xi, size=target_size, mode='bilinear', align_corners=True) for xi in x]
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)


# ==========================================
# 5. MASFNet 整体网络架构
# ==========================================
class MASFNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(MASFNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

        # 清除 SAM2 原始无关组件
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        # 冻结并配置 Hiera 主干与 Adapter 微调
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = [Adapter(block) for block in self.encoder.blocks]
        self.encoder.blocks = nn.Sequential(*blocks)

        # 阶段一：多尺度感知增强模块 (MSPE)
        self.mspe1 = MSPE(144, 64)
        self.mspe2 = MSPE(288, 64)
        self.mspe3 = MSPE(576, 64)
        self.mspe4 = MSPE(1152, 64)

        # 阶段二：双重跨层级特征融合模块 (DCFF)
        self.dcff = DCFF(features=[64, 64, 64], strides=[2, 2, 2])

        # 阶段三：自顶向下解码引导层 (包含 CSGF)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)

        # 侧边监督与最终输出预测头
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Hiera 骨干网络前向传播
        x1, x2, x3, x4 = self.encoder(x)

        orig_size_x1 = x1.size()[2:]
        orig_size_x2 = x2.size()[2:]
        orig_size_x3 = x3.size()[2:]

        # 1. MSPE 单层多尺度提纯
        x1 = self.mspe1(x1)
        x2 = self.mspe2(x2)
        x3 = self.mspe3(x3)
        x4 = self.mspe4(x4)

        # 2. DCFF 跨层空间与通道注意力对齐
        target_size = (88, 88)
        x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=True)
        x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=True)
        x3_resized = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=True)

        x1_dcff, x2_dcff, x3_dcff = self.dcff([x1_resized, x2_resized, x3_resized])

        # 恢复空间分辨率
        x1_aligned = F.interpolate(x1_dcff, size=orig_size_x1, mode='bilinear', align_corners=True)
        x2_aligned = F.interpolate(x2_dcff, size=orig_size_x2, mode='bilinear', align_corners=True)
        x3_aligned = F.interpolate(x3_dcff, size=orig_size_x3, mode='bilinear', align_corners=True)

        # 3. CSGF 逐级自顶向下引导聚合解码
        x = self.up1(x4, x3_aligned)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear', align_corners=True)

        x = self.up2(x, x2_aligned)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear', align_corners=True)

        x = self.up3(x, x1_aligned)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear', align_corners=True)

        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = MASFNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(f"Pred shape: {out.shape}, Side1 shape: {out1.shape}, Side2 shape: {out2.shape}")