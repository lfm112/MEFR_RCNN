import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor
# from collections import OrderedDict
import torch.nn.functional as F
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CIAM(nn.Module):
    def __init__(self,in_channels):
        super(CIAM, self).__init__()
        self.convi_1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.convv_1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.conv_iv_1 = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0)
        self.conv_3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.sam_i = SpatialAttention()
        self.sam_v = SpatialAttention()

        self.cam_i = ChannelAttention(in_channels)
        self.cam_v = ChannelAttention(in_channels)






    def forward(self, fi,fv):
        i = self.convi_1(fi)
        v = self.convv_1(fv)
        iv = i*v

        fi_ = self.cam_i(fi*self.sam_i(iv))*fi
        fv_ = self.cam_v(fv*self.sam_v(iv))*fv
        c = torch.cat((fi_,fv_),1)
        o = self.conv_3(self.conv_iv_1(c)+fv_+fi_)
        return o,fi_,fv_

class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out



class SE_BasicBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(SE_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.SE = SE_Block(planes)           # Squeeze-and-Excitation block

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MAEM(nn.Module):
    def __init__(self,in_channels):
        super(MAEM ,self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels,in_channels,1,1,0)
        self.conv1_2 = nn.Conv2d(3*in_channels,in_channels,1,1,0)
        self.conv1_3 = nn.Conv2d(in_channels,in_channels,1,1,0)
        self.conv3 = nn.Conv2d(in_channels,in_channels,3,1,1)
        self.conv5 = nn.Conv2d(in_channels,in_channels,5,1,2)
        self.SE = SE_BasicBlock(3*in_channels,3*in_channels)
        self.RELU = nn.ReLU()
    def forward(self,x):
        Q_x = torch.cat((self.conv1_1(x),self.conv3(x),self.conv5(x)),dim=1)
        Q_x_ = self.conv1_2(self.SE(Q_x))

        Q = self.RELU(self.conv1_3(Q_x_))
        return Q


def perceptual_loss(F, Y):
    """
    直接计算两个张量之间的感知损失。

    参数:
    - F: Tensor, 预测结果.
    - Y: Tensor, 目标结果.

    返回:
    - loss: Tensor, 损失值.
    """
    # 确保输入张量的形状相同
    if F.shape != Y.shape:
        raise ValueError("输入张量F和Y的形状必须相同.")

    # 计算每个对应元素的平方差，然后在所有维度上求平均
    loss = torch.nn.functional.mse_loss(F, Y, reduction='mean')

    return loss


class CIAFNet(nn.Module):
    def __init__(self, vin_channels=3, iin_channels=1):
        super(CIAFNet, self).__init__()
        self.vlayer1 = nn.Sequential(
            nn.Conv2d(vin_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.ilayer1 = nn.Sequential(
            nn.Conv2d(iin_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.vlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )

        self.ilayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )

        self.vlayer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.ilayer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.vlayer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        )
        self.ilayer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
        )

        self.vlayer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),

        )
        self.ilayer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),

        )
        self.CIAM2 = CIAM(64)
        self.CIAM3 = CIAM(128)
        self.CIAM4 = CIAM(256)
        self.MAEMI = MAEM(512)
        self.MAEMV = MAEM(512)

        self.Delayer1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.Delayer2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.Delayer3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.Delayer4 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)

    def forward(self, I, V):


        I = self.ilayer1(I)
        V = self.vlayer1(V)

        I2 = self.ilayer2(I)
        V2 = self.vlayer2(V)
        IV2, I2_, V2_ = self.CIAM2(I2, V2)

        I3 = self.ilayer3(I2 + I2_)
        V3 = self.vlayer3(V2 + V2_)
        IV3, I3_, V3_ = self.CIAM3(I3, V3)

        I4 = self.ilayer4(I3 + I3_)
        V4 = self.vlayer4(V3 + V3_)
        IV4, I4_, V4_ = self.CIAM4(I4, V4)

        I5 = self.ilayer5(I4 + I4_)
        V5 = self.vlayer5(V4 + V4_)
        F = self.MAEMI(I5) + self.MAEMV(V5)

        F = self.Delayer1(F) + IV4
        F = self.Delayer2(F) + IV3
        F = self.Delayer3(F) + IV2
        F = self.Delayer4(F)
        L_rec =( torch.abs(F-I)+torch.abs(F-V)).mean()
        L_perc = 0.5*perceptual_loss(F,I)+0.5*perceptual_loss(F,V)



        return F,L_rec,L_perc














































def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)





class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(32,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)
        # self.blocks = nn.ModuleList(blocks)
        out_channels = 256
        self.out_channels = out_channels
        self.conv_x1 = MBConv(input_c=192,out_c=256,stride=1,se_ratio=0.25,drop_rate=0.1,kernel_size=3,expand_ratio=4,norm_layer=norm_layer)
        self.conv_x2 = MBConv(input_c=224,out_c=256,stride=1,se_ratio=0.25,drop_rate=0.1,kernel_size=3,expand_ratio=6,norm_layer=norm_layer)
        self.conv_x3 = MBConv(input_c=384,out_c=256,stride=1,se_ratio=0.25,drop_rate=0.1,kernel_size=3,expand_ratio=6,norm_layer=norm_layer)
        self.eca_layer = eca_layer(96)
        self.fusion_net = CIAFNet()

    def forward(self, x):



        VIS = x[:, :3, :, :]
        I = x[:, 3, :, :].unsqueeze(1)




        x,L_rec,L_perc = self.fusion_net(I,VIS)



        # print(self.blocks)
        x = self.stem(x)


        # ------------------------------------------------------
        for i in range(2):
            x = self.blocks[i](x)


        # ------------------------------------------------------
        for i in range(4):
            x = self.blocks[i+2](x)


        # ------------------------------------------------------
        for i in range(3):
            x = self.blocks[i+6](x)


        x = self.eca_layer(x)

        # ------------------------------------------------------
        for i in range(6):
            x = self.blocks[i+9](x)

        x1 = self.conv_x1(x)

        # ------------------------------------------------------
        for i in range(6):
            x = self.blocks[i+15](x)

        x2 = self.conv_x2(x)

        # ------------------------------------------------------
        for i in range(8):
            x = self.blocks[i+21](x)

        x3 = self.conv_x3(x)
        ordered_dict = OrderedDict()
        ordered_dict['0'] = x1
        ordered_dict['1'] = x2
        ordered_dict['2'] = x3





        return ordered_dict,L_rec,L_perc


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

def mefr_model(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 32, 32, 0, 0],
                    [4, 3, 2, 4, 32, 64, 0, 0],
                    [3, 3, 2, 4, 64, 96, 0, 0],
                    [6, 3, 2, 4, 96, 192, 1, 0.25],
                    [6, 3, 2, 6, 192, 224, 1, 0.25],
                    [8, 3, 2, 6, 224, 384, 1, 0.25],
                    ]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model
# model = zts_model()
# # model = efficientnetv2_l()
# # print(model)
# x = torch.randn(2,3,800,1216)
# x = model(x)



# 假设你想提取每个 stage 最后一层的输出



# class ztsnet(nn.Module):
#     def __init__(self,
#                  model_cnf: list,
#                  num_classes: int = 1000,
#                  num_features: int = 1280,
#                  dropout_rate: float = 0.2,
#                  drop_connect_rate: float = 0.2):
#         super(ztsnet, self).__init__()
#         self.stem = ConvBNAct(3,
#                               32,
#                               kernel_size=3,
#                               stride=2,
#                               norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1))  # 激活函数默认是SiLU
#         self.layer1 = nn.Sequential(FusedMBConv(kernel_size=3,input_c=32,out_c=32))
#
#
#
#     def forward(self, x):
#         x = self.stem(x)
#         print(x.shape)
# z = ztsnet()