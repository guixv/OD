import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Optional


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, out_channel=None, active_layer=nn.GELU, drop_pro=0.):
        super(Mlp, self).__init__()
        out_channel = out_channel or in_channel
        hidden_channel = hidden_channel or in_channel
        self.fc1 = nn.Linear(in_features=in_channel, out_features=hidden_channel)
        self.act = active_layer()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        self.dropout = nn.Dropout(drop_pro)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channel=3, patch_size=4, embed_dim=128, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.in_channel = in_channel
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(self.in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        if (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0):
            # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
            x = functional.pad(x, (
                0, self.patch_size[1] - W % self.patch_size[1], 0, self.patch_size[0] - H % self.patch_size[0], 0, 0))

        x = self.proj(x)
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.contiguous()
    x = x.view(-1, window_size, window_size, C)
    return x


def window_reverse(window, window_size, H, W):
    """
        将一个个window还原成一个feature map
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int,int): Window size(M)
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
    B = int(window.shape[0] / ((H * W) / (window_size[0] * window_size[1])))
    x = window.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_head, qkv_bias=True, atten_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5
        self.relative_position_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_head)
        )
        self.relative_position_indexs = self.relative_position_indexs(window_size)
        self.register_buffer("relative_position_index", self.relative_position_indexs)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def relative_position_indexs(self, window_size):
        h = torch.arange(window_size[0])
        w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([h, w], indexing='ij'))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1, -1)  # [2, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        return relative_position_index

    def forward(self, x, mask: Optional[torch.Tensor] = None):  # Optional??
        BN, N, C = x.shape  # [batch_size*num_window,Mh*Mw, total_embed_dim]
        x = self.qkv(x)
        x = x.reshape(BN, N, 3, self.num_head, C // self.num_head)
        x = x.permute(2, 0, 3, 1, 4)
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = x.unbind(0)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]

        k = k.transpose(-1, -2)
        atten = (q @ k) * self.scale
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]

        relative_position_bias = self.relative_position_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        atten = atten + relative_position_bias.unsqueeze(0)

        atten = self.softmax(atten)
        # mask unsolved

        atten = self.atten_drop(atten)

        x = atten @ v
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        x = x.transpose(1, 2)
        # transpose: [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = x.reshape(BN, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_head, qkv_bias=True, drop_rate=0., act_layer=nn.GELU,
                 atten_drop=0.0, path_drop=0., mlp_ratio=4.0, norm_layer=nn.LayerNorm, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_head = num_head
        self.proj_drop = drop_rate
        self.shift_size=shift_size
        self.atten_drop = atten_drop
        self.path_drop = path_drop
        self.act = act_layer
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size[0], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.shift_size = shift_size
        self.atten = WindowAttention(dim=dim, window_size=self.window_size, num_head=num_head, qkv_bias=qkv_bias,
                                     atten_drop=atten_drop, proj_drop=self.proj_drop)

        self.drop_path = DropPath(self.path_drop) if self.path_drop > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_channel=dim, hidden_channel=mlp_hidden_dim, active_layer=self.act, drop_pro=self.proj_drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size| L != H * W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size[0])  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.atten(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()  # functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            # [B, H, W, C]

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)  # res
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, depth, num_head, window_size=7, qkv_bias=True, drop_rate=0., act_layer=nn.GELU,
                 atten_drop=0.0, path_drop=0., mlp_ratio=4.0, norm_layer=nn.LayerNorm, downsample=None):

        # downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        super(SwinTransformerLayer, self).__init__()
        self.shift_size = window_size // 2
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.blocks = nn.Sequential(*[SwinTransformerBlock(dim=dim, window_size=window_size,
                                                           num_head=num_head, qkv_bias=qkv_bias,
                                                           drop_rate=drop_rate, act_layer=act_layer,
                                                           atten_drop=atten_drop, path_drop=path_drop,
                                                           mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else self.shift_size)
                                      for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slice = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slice = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

        cnt = 0
        for h in h_slice:
            for w in w_slice:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        window_mask = window_partition(img_mask, self.window_size)
        window_mask = window_mask.view(-1, self.window_size * self.window_size)
        attn_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        atten_mask = self.create_mask(x, H, W)
        for block in self.blocks:
            block.H, block.W = H, W
            x = block(x, atten_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, in_channel=3, patch_size=4,
                 window_size=7, embed_dim=128,
                 depth=(), num_head=(), num_classes=10,
                 norm_layer=nn.LayerNorm, drop_rate=0.,
                 atten_drop=0.0, drop_path_rate=0., mlp_ratio=4.,
                 qkv_bias=True, act_layer=nn.GELU,
                 init_weights=True):
        super(SwinTransformer, self).__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.window_siz = window_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_head = num_head
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.num_layer = len(depth)
        self.num_feature = int(embed_dim * 2 ** (self.num_layer - 1))
        self.patch_embed = PatchEmbed(in_channel=self.in_channel,
                                      patch_size=self.patch_size,
                                      embed_dim=self.embed_dim,
                                      norm_layer=self.norm_layer)
        self.mlp_ratio = mlp_ratio
        self.dropout = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule

        self.layers = nn.Sequential(*[SwinTransformerLayer(dim=int(embed_dim * 2 ** i_layer),
                                                           depth=depth[i_layer],
                                                           num_head=num_head[i_layer],
                                                           window_size=window_size,
                                                           qkv_bias=qkv_bias,
                                                           drop_rate=drop_rate,
                                                           act_layer=act_layer,
                                                           atten_drop=atten_drop,
                                                           path_drop=dpr[i_layer],
                                                           norm_layer=norm_layer,
                                                           downsample=PatchMerging if (
                                                                       i_layer < self.num_layer - 1) else None, )
                                      for i_layer in range(self.num_layer)
                                      ])
        self.norm = norm_layer(self.num_feature)
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_feature, num_classes) if num_classes > 0 else nn.Identity()
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.dropout(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B,H*W,C]
        x = self.avgPool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_transformer(num_classes=10):
    module = SwinTransformer(in_channel=3,
                             patch_size=4,
                             window_size=7,
                             embed_dim=128,
                             depth=(2, 2, 18, 2),
                             num_head=(4, 8, 16, 32),
                             num_classes=num_classes)
    return module
