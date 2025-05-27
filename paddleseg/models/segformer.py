# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers, backbones
from paddleseg.utils import utils


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # Transposed convolution
        self.transconv_c4 = nn.Conv2DTranspose(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=8,
                                               stride=8, padding=0)
        self.transconv_c3 = nn.Conv2DTranspose(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=6,
                                               stride=4, padding=1)
        self.transconv_c2 = nn.Conv2DTranspose(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=4,
                                               stride=2, padding=1)
        self.transconv_logit = nn.Conv2DTranspose(in_channels=num_classes, out_channels=num_classes, kernel_size=6,
                                                  stride=4, padding=1)
        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)  # 4x downsampling
        c2_shape = paddle.shape(c2)  # 8x downsampling
        c3_shape = paddle.shape(c3)  # 16x downsampling
        c4_shape = paddle.shape(c4)  # 32x downsampling

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        # 8x upsampling
        # _c4 = F.interpolate(
        #     _c4,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        _c4 = self.transconv_c4(_c4)
        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        # 4x upsampling
        # _c3 = F.interpolate(
        #     _c3,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        _c3 = self.transconv_c3(_c3)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        # 2x upsampling
        # _c2 = F.interpolate(
        #     _c2,
        #     size=c1_shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        _c2 = self.transconv_c2(_c2)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))
        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        logit = self.transconv_logit(logit)
        # 4x upsampling
        # return [
        #     F.interpolate(
        #         logit,
        #         size=paddle.shape(x)[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]
        return [logit]


@manager.MODELS.add_component
class SegFormer_ori(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer_ori, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(in_channels=embedding_dim * 4,
                                             out_channels=embedding_dim,
                                             kernel_size=1,
                                             bias_attr=False)

        self.linear_pred = nn.Conv2D(embedding_dim,
                                     self.num_classes,
                                     kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = c1.shape
        c2_shape = c2.shape
        c3_shape = c3.shape
        c4_shape = c4.shape

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(_c4,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(_c3,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(_c2,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(logit,
                          size=x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
        ]

# if __name__ == '__main__':
#     arr = paddle.rand((1, 1, 1024, 1024))
#     model = SegFormer(num_classes=2, backbone=backbones.MixVisionTransformer_B5(in_channels=1), embedding_dim=768)
#     # ans = model(arr)
#     paddle.summary(model, input_size=(1, 1, 1024, 1024))
