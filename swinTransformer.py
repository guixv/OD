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
    def __init__(self, in_channel=3, patch_size=4, embed_dim=128, norm_layer=nn.LayerNorm):
        super(PatchEmbed, self).__init__()
        self.in_channel = in_channel
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.norm = norm_layer
        self.proj = nn.Conv2d(self.in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4)
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
        head_dim = self.dim // self.num_head
        self.scale = head_dim ** -0.5
        self.relative_position_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_head)
        )
        self.relative_position_index = self.relative_position_index(window_size)
        self.register_buffer("relative_position_index", self.relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def relative_position_index(self, window_size):
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
        self.atten_drop = atten_drop
        self.path_drop = path_drop
        self.act = act_layer
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer
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
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size
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
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
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
                 atten_drop=0.0, path_drop=0., mlp_ratio=4.0, norm_layer=nn.LayerNorm, shift_size=0,downsample=None):

        #downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
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

    def create_mask(self,x,H,W):


class SwinTransformer(nn.Module):
    def __init__(self, in_channel=3, patch_size=4,
                 window_size=7, embed_dim=128,
                 depth=(), num_head=(), num_classes=10,
                 norm_layer=nn.LayerNorm, drop_rate=0.,
                 atten_drop=0.0, path_drop=0., mlp_ratio=4.):
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


def swinTransformer(num_classes=10):
    module = SwinTransformer(in_channel=3,
                             patch_size=4,
                             window_size=7,
                             embed_dim=128,
                             depth=(2, 2, 18, 2),
                             num_head=(4, 8, 16, 32),
                             num_classes=num_classes)
    return module
