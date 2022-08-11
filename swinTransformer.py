import torch
import torch.nn as nn
import torch.nn.functional as functional


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
            x = functional.pad(x, (0, W % self.patch_size[1], 0, H % self.patch_size[0], 0, 0))

        x = self.proj(x)
        B, C, H, W = x.shape
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


class SwinTransformer(nn.Module):
    def __init__(self, in_channel=3, patch_size=4,
                 window_size=7, embed_dim=128,
                 depth=(), num_head=(), num_classes=10,
                 norm_layer=nn.LayerNorm, drop_rate=0., mlp_ratio=4.):
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
