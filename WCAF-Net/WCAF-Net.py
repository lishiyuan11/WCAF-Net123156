import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from libs.backbone.pvtv2 import pvt_v2_b2
import pywt
# pip install pywavelets==1.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple
# ================= 基础模块 =================
def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size, stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))

class ResidualBlock(nn.Module):
    """ Conv+BN+ReLU+Conv+BN + 残差连接 + ReLU """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))  # 计算用于位置编码的角度
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))  # 计算衰减因子
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay)  # 生成高度索引
        index_w = torch.arange(W).to(self.decay)  # 生成宽度索引
        grid = torch.meshgrid([index_h, index_w])  # 创建索引网格
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 将网格转换为 (H*W, 2) 的形状
        mask = grid[:, None, :] - grid[None, :, :]  # 计算曼哈顿距离
        mask = (mask.abs()).sum(dim=-1)  # 对最后一个维度求和
        mask = mask * self.decay[:, None, None]  # 将距离与衰减因子相乘
        return mask

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)  # 生成一维索引
        mask = index[:, None] - index[None, :]  # 计算一维曼哈顿距离
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]  # 将距离与衰减因子相乘
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))  # 计算正弦位置编码
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))  # 计算余弦位置编码
            retention_rel_pos = ((sin, cos), self.decay.exp())  # 结合衰减因子

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)  # 生成像素位置索引
            sin = torch.sin(index[:, None] * self.angle[None, :])  # 计算正弦位置编码
            sin = sin.reshape(slen[0], slen[1], -1)  # 重塑为三维张量

            cos = torch.cos(index[:, None] * self.angle[None, :])  # 计算余弦位置编码
            cos = cos.reshape(slen[0], slen[1], -1)  # 重塑为三维张量

            mask_h = self.generate_1d_decay(slen[0])  # 生成H方向的衰减掩码
            mask_w = self.generate_1d_decay(slen[1])  # 生成W方向的衰减掩码

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))  # 组合位置编码与衰减掩码

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])  # 生成二维衰减掩码
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]  # 取出偶数维度
    x2 = x[:, :, :, :, 1::2]  # 取出奇数维度
    x = torch.stack([-x2, x1], dim=-1)  # 旋转操作
    out = x.flatten(-2)  # 展平
    return out

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)  # 调制特征的相对位置

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)  # 转置为 (b c h w)
        x = self.conv(x)  # 执行卷积操作
        x = x.permute(0, 2, 3, 1)  # 转置回 (b h w c)
        return x

class VisionRetentionChunk(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos  # 提取位置编码、掩码

        q = self.q_proj(x)  # 计算查询向量
        k = self.k_proj(x)  # 计算键向量
        v = self.v_proj(x)  # 计算值向量

        lepe = self.lepe(v)  # 执行局部增强卷积 最后一步进行叠加

        k *= self.scaling  # 缩放键向量
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # 调整查询向量形状
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # 调整键向量形状
        qr = theta_shift(q, sin, cos)  # 应用位置调制
        kr = theta_shift(k, sin, cos)  # 应用位置调制

        qr_w = qr.transpose(1, 2)  # 转置以便在宽度方向上计算
        kr_w = kr.transpose(1, 2)  # 转置以便在宽度方向上计算
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # 调整值向量形状
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # 计算宽度方向的注意力
        qk_mat_w = qk_mat_w + mask_w  # 加上宽度方向的掩码
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # 归一化
        v = torch.matmul(qk_mat_w, v)  # 应用注意力权重

        qr_h = qr.permute(0, 3, 1, 2, 4)  # 转置以便在高度方向上计算
        kr_h = kr.permute(0, 3, 1, 2, 4)  # 转置以便在高度方向上计算
        v = v.permute(0, 3, 2, 1, 4)  # 调整值向量形状
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # 计算高度方向的注意力
        qk_mat_h = qk_mat_h + mask_h  # 加上高度方向的掩码
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # 归一化
        output = torch.matmul(qk_mat_h, v)  # 应用注意力权重

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # 调整输出形状

        output = output + lepe  # 加上局部增强的输出
        output = self.out_proj(output)  # 线性投影输出
        return output
# ================= ASFF 模块 =================
class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512, compress_c=8):
        super(ASFF_4, self).__init__()
        self.inter_dim = inter_dim

        # 压缩卷积
        self.weight_level_0 = BasicConv(inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(inter_dim, compress_c, 1, 1)

        # 生成 1 通道权重
        self.weight0 = nn.Conv2d(compress_c, 1, kernel_size=1)
        self.weight1 = nn.Conv2d(compress_c, 1, kernel_size=1)
        self.weight2 = nn.Conv2d(compress_c, 1, kernel_size=1)
        self.weight3 = nn.Conv2d(compress_c, 1, kernel_size=1)

        # 融合后的卷积
        self.conv = BasicConv(inter_dim, inter_dim, 3, 1)

    def forward(self, x1, x2, x3, x4):
        # 压缩
        l0 = self.weight_level_0(x1)
        l1 = self.weight_level_1(x2)
        l2 = self.weight_level_2(x3)
        l3 = self.weight_level_3(x4)

        # 得到 1 通道权重
        w0 = self.weight0(l0)
        w1 = self.weight1(l1)
        w2 = self.weight2(l2)
        w3 = self.weight3(l3)

        # 加权
        out1 = x1 * w0
        out2 = x2 * w1
        out3 = x3 * w2
        out4 = x4 * w3

        # 上采样到统一尺度
        target_size = x1.size()[2:]
        out2 = F.interpolate(out2, size=target_size, mode="bilinear", align_corners=True)
        out3 = F.interpolate(out3, size=target_size, mode="bilinear", align_corners=True)
        out4 = F.interpolate(out4, size=target_size, mode="bilinear", align_corners=True)

        fused = out1 + out2 + out3 + out4
        fused = self.conv(fused)  # 融合卷积
        return fused

# ================= 多尺度特征融合 FF =================
class FF(nn.Module):
    def __init__(self, channels=[64, 128, 320, 512], out_dim=64):
        super(FF, self).__init__()
        # 多尺度卷积
        self.conv1 = BasicConv(channels[0], channels[0], 1)
        self.conv2 = BasicConv(channels[1], channels[1], 3)
        self.conv3 = BasicConv(channels[2], channels[2], 5)
        self.conv4 = BasicConv(channels[3], channels[3], 7)

        # 统一通道到一个中间维度
        self.reduce1 = nn.Conv2d(channels[0], out_dim, 1, bias=False)
        self.reduce2 = nn.Conv2d(channels[1], out_dim, 1, bias=False)
        self.reduce3 = nn.Conv2d(channels[2], out_dim, 1, bias=False)
        self.reduce4 = nn.Conv2d(channels[3], out_dim, 1, bias=False)

        # ASFF 融合
        self.asff = ASFF_4(inter_dim=out_dim)

        # 残差块
        self.res_block = ResidualBlock(out_dim)

        # 最终降维
        self.out_conv = nn.Conv2d(out_dim, out_dim // 2, 1, bias=False)

    def forward(self, x1, x2, x3, x4):
        # 多尺度卷积
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # 通道统一
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)

        # 融合
        fused = self.asff(x1, x2, x3, x4)

        # 残差块
        fused = self.res_block(fused)

        # 降维输出
        out = self.out_conv(fused)
        return out

class High(nn.Module):
    def __init__(self, channels):
        super(High, self).__init__()
        # 第一个 depthwise 卷积
        self.dw_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pw_conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 第二个 depthwise 卷积
        self.dw_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pw_conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 激活和归一化：关键修改！
        self.gelu = nn.GELU()
        # 1. 修正 normalized_shape：permute后通道在最后1维，故只需指定 [channels]（对应最后1维）
        self.norm = nn.LayerNorm([channels])  # 仅对通道维度归一化，无需指定H/W

    def forward(self, x):
        # 第一个残差分支
        residual1 = x
        out = self.dw_conv1(x)
        out = self.pw_conv1(out)
        out = out + residual1
        out = self.gelu(out)

        # 第二个残差分支
        residual2 = out
        out = self.dw_conv2(out)
        out = self.pw_conv2(out)
        out = out + residual2

        # LayerNorm：关键修改！
        b, c, h, w = out.shape
        out = out.permute(0, 2, 3, 1)  # BCHW -> BHWC（通道到最后1维）
        # 2. 此时输入形状为 (b, h, w, c)，normalized_shape=[c] 正好匹配最后1维
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # BHWC -> BCHW（恢复原维度顺序）

        # 最外层残差
        out = out + x
        return out

# 定义一个DWT功能类，继承自Function
class DWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保证输入张量x在内存中是连续存储的
        x = x.contiguous()
        # 保存后向传播需要的参数
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 获取输入张量x的通道数
        dim = x.shape[1]
        # 对x进行二维卷积操作，得到低频和高频分量
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个分量按通道维度拼接起来
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回拼接后的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的权重
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            # 调整dx的维度顺序并重塑
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 将四个小波滤波器沿零维度拼接，并重复C次以匹配输入通道数
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 使用转置卷积进行上采样
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        # 返回dx以及其余不需要梯度的参数
        return dx, None, None, None, None

# 定义一个二维离散小波变换模块，继承自nn.Module
class DWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建分解低通和高通滤波器的Tensor
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # 计算二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # 注册缓冲区变量来存储滤波器
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # 确保滤波器的数据类型为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用DWT_Function的forward方法
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 定义一个IDWT功能类，继承自Function
class IDWT_Function(Function):
    # 定义前向传播静态方法
    @staticmethod
    def forward(ctx, x, filters):
        # 保存后向传播需要的参数
        ctx.save_for_backward(filters)
        # 保存输入张量x的形状
        ctx.shape = x.shape

        # 根据保存的形状信息调整x的形状
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # 计算通道数
        C = x.shape[1]
        # 重塑x
        x = x.reshape(B, -1, H, W)
        # 重复滤波器C次以匹配输入通道数
        filters = filters.repeat(C, 1, 1, 1)
        # 使用转置卷积进行上采样
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        # 返回上采样的结果
        return x

    # 定义反向传播静态方法
    @staticmethod
    def backward(ctx, dx):
        # 检查是否需要计算x的梯度
        if ctx.needs_input_grad[0]:
            # 取出前向传播时保存的滤波器
            filters = ctx.saved_tensors
            filters = filters[0]
            # 根据保存的形状信息重塑dx
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            # 分解滤波器
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 对dx进行二维卷积操作，得到低频和高频分量
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 将四个分量按通道维度拼接起来
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # 返回dx以及其余不需要梯度的参数
        return dx, None

# 定义一个二维逆离散小波变换模块，继承自nn.Module
class IDWT_2D(nn.Module):
    # 初始化函数，接受一个小波基名称作为参数
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 使用pywt库创建指定的小波对象
        w = pywt.Wavelet(wave)
        # 创建重构低通和高通滤波器的Tensor
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # 为滤波器添加额外的维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        # 将四个小波滤波器沿零维度拼接
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册缓冲区变量来存储滤波器
        self.register_buffer('filters', filters)
        # 确保滤波器的数据类型为float32
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播函数
    def forward(self, x):
        # 应用IDWT_Function的forward方法
        return IDWT_Function.apply(x, self.filters)

class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        out2 = out2 + x # Residual connection
        return out2


class Fusion(nn.Module):
    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)

        # 通道调整层（关键新增）
        self.adjust_low = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)  # in_channels→64
        self.adjust_high = nn.Conv2d(3 * in_channels, 64, kernel_size=1, bias=False)  # 3*in_channels→64

        self.high = High(64)  # 高频处理模块输入通道=64
        self.pos = RetNetRelPos2d(embed_dim=64, num_heads=4, initial_value=1, heads_range=3)
        self.Model = VisionRetentionChunk(embed_dim=64, num_heads=4)  # 与64通道匹配

    def forward(self, x):
        b, c, h, w = x.shape  # c应为in_channels（如96）
        x_dwt = self.dwt(x)  # 输出通道数=4*c

        # 拆分DWT结果（每个分量通道数=c）
        ll, lh, hl, hh = x_dwt.split(c, 1)


        # 处理高频特征（调整通道至64）
        high = torch.cat([lh, hl, hh], 1)  # 通道数=3*c
        high = self.adjust_high(high)  # 3*c→64
        highf = self.high(high)  # (B, 64, H, W)

        # 拼接并逆变换
        out = torch.cat((ll, highf), 1)  # (B, 128, H, W)，128是4的倍数
        out_idwt = self.idwt(out)
        return out_idwt
class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAwareFeatureEnhancer, self).__init__()
        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        edge_feature = x-self.edge_extractor(x)
        edge_weights = self.weight_generator(edge_feature)
        enhanced_features = edge_weights*x +x
        return enhanced_features
# ================= 主网络 =================
class WCOD(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super(WCOD, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        if imagenet_pretrained:
            path = './pth/backbone/pvt_v2_b2.pth'
            save_model = torch.load(path, map_location="cpu")
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        self.ff = FF()
        self.conv32_1 = nn.Conv2d(24,1,1)
        self.fusion = Fusion(32, wave='haar')
        self.edge = EdgeAwareFeatureEnhancer(24)
    def forward(self, x):
        pvt = self.backbone(x)  # 返回 [x1,x2,x3,x4]
        x1, x2, x3, x4 = pvt
        out = self.ff(x1, x2, x3, x4)
        out = self.fusion(out)
        out = self.edge(out)
        out = self.conv32_1(out)
        out = F.interpolate(out,size=(256, 256),mode='bilinear')

        return out

# ================= 测试 =================
if __name__ == "__main__":
    net = WCOD(imagenet_pretrained=False)
    net.eval()
    dump_x = torch.randn(2, 3, 256,256)
    y = net(dump_x)
    print("输出特征尺寸:", y.shape)
