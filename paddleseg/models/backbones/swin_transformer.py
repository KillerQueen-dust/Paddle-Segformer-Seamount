# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.utils import utils
from paddleseg.models.backbones.transformer_utils import *


class Mlp(nn.Layer):
    """ Multilayer perceptron."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4,
                           5]).reshape([-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x


class WindowAttention(nn.Layer):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                   num_heads),
            default_initializer=zeros_)
        self.add_parameter("relative_position_bias_table",
                           self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h,
                                               coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2

        relative_coords = relative_coords.transpose([1, 2, 0])

        relative_coords[:, :, 0] += self.window_size[
                                        0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            [B_, N, 3, self.num_heads,
             C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        index = self.relative_position_index.reshape([-1])
        relative_position_bias = paddle.index_select(
            self.relative_position_bias_table, index)

        relative_position_bias = relative_position_bias.reshape([
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        ])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N
                                 ]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    """
    Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        x = x.transpose([0, 3, 1, 2])
        x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
        x = x.transpose([0, 2, 3, 1])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size,
             C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp,
                                   Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                axis=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Layer):
    """
    Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape([B, H, W, C])

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = x.transpose([0, 3, 1, 2])
            x = F.pad(x, [0, W % 2, 0, H % 2])
            x = x.transpose([0, 2, 3, 1])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Layer):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = paddle.zeros((1, Hp, Wp, 1))  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.reshape(
            [-1, self.window_size * self.window_size])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        huns = -100.0 * paddle.ones_like(attn_mask)
        attn_mask = huns * (attn_mask != 0).astype("float32")

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Layer):
    """
    Image to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, [0, self.patch_size[1] - W % self.patch_size[1], 0, 0])
        if H % self.patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            _, _, Wh, Ww = x.shape
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm(x)
            x = x.transpose([0, 2, 1]).reshape([-1, self.embed_dim, Wh, Ww])

        return x


@manager.BACKBONES.add_component
class SwinTransformer(nn.Layer):
    """
    The SwinTransformer implementation based on PaddlePaddle.

    The original article refers to
    Liu, Ze, et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    (https://arxiv.org/abs/2103.14030)

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model, used in absolute postion embedding. Default: 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters. Default: -1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 pretrained=None):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1]
            ]

            self.absolute_pos_embed = self.create_parameter(
                shape=(1, embed_dim, patches_resolution[0],
                       patches_resolution[1]),
                default_initializer=zeros_)
            self.add_parameter("absolute_pos_embed", self.absolute_pos_embed)
            trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        feat_channels = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.feat_channels = feat_channels

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(feat_channels[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_sublayer(layer_name, layer)

        self._freeze_stages()

        self.pretrained = pretrained
        self.init_weights(self.pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                layer = self.layers[i]
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Linear):
                    trunc_normal_(sublayer.weight)
                    if isinstance(sublayer,
                                  nn.Linear) and sublayer.bias is not None:
                        zeros_(sublayer.bias)
                elif isinstance(sublayer, nn.LayerNorm):
                    zeros_(sublayer.bias)
                    ones_(sublayer.weight)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        _, _, Wh, Ww = x.shape
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose([0, 2, 1])  # B Wh*Ww C

        else:
            x = x.flatten(2).transpose([0, 2, 1])
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.reshape(
                    [-1, H, W, self.feat_channels[i]]).transpose([0, 3, 1, 2])
                outs.append(out)

        return tuple(outs)

    def train(self):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train()
        self._freeze_stages()


@manager.BACKBONES.add_component
class SwinTransformer_tiny_patch4_window7_224_maskformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=224,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            patch_norm=True,
            **kwargs)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.feat_channels[0],
            "res3": self.feat_channels[1],
            "res4": self.feat_channels[2],
            "res5": self.feat_channels[3],
        }

    def forward(self, x):
        outputs = {}
        y = super(SwinTransformer_tiny_patch4_window7_224_maskformer,
                  self).forward(x)
        for i, k in enumerate(self._out_features):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: {
                "channels": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name]
            }
            for name in self._out_features
        }


@manager.BACKBONES.add_component
def SwinTransformer_tiny_patch4_window7_224(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs)

    return model


@manager.BACKBONES.add_component
class SwinTransformer_small_patch4_window7_224_maskformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=224,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            patch_norm=True,
            **kwargs)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.feat_channels[0],
            "res3": self.feat_channels[1],
            "res4": self.feat_channels[2],
            "res5": self.feat_channels[3],
        }

    def forward(self, x):
        outputs = {}
        y = super(SwinTransformer_small_patch4_window7_224_maskformer,
                  self).forward(x)
        for i, k in enumerate(self._out_features):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: {
                "channels": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name]
            }
            for name in self._out_features
        }


@manager.BACKBONES.add_component
def SwinTransformer_small_patch4_window7_224(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        **kwargs)

    return model


@manager.BACKBONES.add_component
def SwinTransformer_base_patch4_window7_224(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=224,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        **kwargs)

    return model


@manager.BACKBONES.add_component
class SwinTransformer_base_patch4_window7_384_maskformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.3,
            patch_norm=True,
            **kwargs)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.feat_channels[0],
            "res3": self.feat_channels[1],
            "res4": self.feat_channels[2],
            "res5": self.feat_channels[3],
        }

    def forward(self, x):
        outputs = {}
        y = super(SwinTransformer_base_patch4_window7_384_maskformer,
                  self).forward(x)
        for i, k in enumerate(self._out_features):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: {
                "channels": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name]
            }
            for name in self._out_features
        }


@manager.BACKBONES.add_component
def SwinTransformer_base_patch4_window12_384(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        **kwargs)

    return model


@manager.BACKBONES.add_component
def SwinTransformer_large_patch4_window7_224(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=224,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs)

    return model


@manager.BACKBONES.add_component
class SwinTransformer_large_patch4_window7_384_maskformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=384,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            drop_path_rate=0.3,
            patch_norm=True,
            **kwargs)

        self._out_features = ["res2", "res3", "res4", "res5"]

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.feat_channels[0],
            "res3": self.feat_channels[1],
            "res4": self.feat_channels[2],
            "res5": self.feat_channels[3],
        }

    def forward(self, x):
        outputs = {}
        y = super(SwinTransformer_large_patch4_window7_384_maskformer,
                  self).forward(x)
        for i, k in enumerate(self._out_features):
            outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {
            name: {
                "channels": self._out_feature_channels[name],
                "stride": self._out_feature_strides[name]
            }
            for name in self._out_features
        }


@manager.BACKBONES.add_component
def SwinTransformer_large_patch4_window12_384(**kwargs):
    model = SwinTransformer(
        pretrain_img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        **kwargs)

    return model


if __name__ == '__main__':
    x = paddle.randn((1, 3, 224, 224))
    model = SwinTransformer_base_patch4_window7_224(ape=True)
    paddle.summary(model, input_size=(1, 3, 224, 224))
