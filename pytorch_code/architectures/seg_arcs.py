import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as t_F
import scipy.io as sio
from math import sqrt


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Block, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        return out


class Seg(nn.Module): #depth = 8

    def __init__(self, input_channs, output_channs, stride=1):
        super(Seg, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride)
        self.conv2 = Block(64, 128, stride)
        self.conv3 = Block(128, 256, stride)
        self.conv4 = Block(256, 512, stride)
        self.conv5 = Block(512, 256, stride)
        self.conv6 = Block(256, 128, stride)
        self.conv7 = Block(128, 64, stride)
        self.conv8 = Block(64, 32, stride)
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv_out(x)

        return x


class BlockBN(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BlockBN, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class SegBN(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(Seg, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride)
        self.conv2 = BlockBN(64, 128, stride)
        self.conv3 = BlockBN(128, 256, stride)
        self.conv4 = BlockBN(256, 512, stride)
        self.conv5 = BlockBN(512, 256, stride)
        self.conv6 = BlockBN(256, 128, stride)
        self.conv7 = BlockBN(128, 64, stride)
        self.conv8 = BlockBN(64, 32, stride)
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv_out(x)

        return x


class BlockBN_pool(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BlockBN_pool, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        out = self.maxpool(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class BlockUpT(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BlockUpT, self).__init__()

        self.up = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2)
        self.conv1 = conv3x3(inplanes + planes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x, enc_feats):

        x = self.up(x)
        [_, _, h, w] = enc_feats.size()
        x = t_F.center_crop(x, [h, w])

        identity = x

        x = torch.cat([x, enc_feats], dim=1)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out


class BlockUp(nn.Module): #residual mode

    def __init__(self, inplanes, planes, stride=1):
        super(BlockUp, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x, enc_feats):

        x = self.up(x)

        identity = x

        diffY = x.size()[2] - enc_feats.size()[2]
        diffX = x.size()[3] - enc_feats.size()[3]
        enc_feats = F.pad(enc_feats, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        x = torch.cat([x, enc_feats], dim=1)

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out


class SegU_t(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(SegU_t, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride) # 1 / 1
        self.conv2 = BlockBN_pool(64, 128, stride) # 1 / 2
        self.conv3 = BlockBN_pool(128, 256, stride) # 1 / 4
        self.conv4 = BlockBN_pool(256, 512, stride) # 1 / 8
        self.conv5 = BlockBN_pool(512, 512, stride)  # 1 / 16

        self.conv1_u = BlockUpT(512, 256, stride) # 1 / 8
        self.conv2_u = BlockUpT(256, 128, stride)  # 1 / 4
        self.conv3_u = BlockUpT(128, 64, stride)  # 1 / 2
        self.conv4_u = BlockUpT(64, 32, stride)  # 1 / 1
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.conv1_u(x5, x4)
        x = self.conv2_u(x, x3)
        x = self.conv3_u(x, x2)
        x = self.conv4_u(x, x1)

        x = self.conv_out(x)

        return x


class SegU(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(SegU_t, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride)  # 1 / 1
        self.conv2 = BlockBN_pool(64, 128, stride)  # 1 / 2
        self.conv3 = BlockBN_pool(128, 256, stride)  # 1 / 4
        self.conv4 = BlockBN_pool(256, 512, stride)  # 1 / 8
        self.conv5 = BlockBN_pool(256, 512, stride)  # 1 / 16
        self.conv1_u = BlockUp(512, 256, stride)  # 1 / 8
        self.conv2_u = BlockUp(256, 128, stride)  # 1 / 4
        self.conv3_u = BlockUp(128, 64, stride)  # 1 / 2
        self.conv4_u = BlockUp(64, 32, stride)  # 1 / 1
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.conv1_u(x5, x4)
        x = self.conv2_u(x, x3)
        x = self.conv3_u(x, x2)
        x = self.conv4_u(x, x1)

        x = self.conv_out(x)

        return x


class ResBlock_conv(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = out + identity


        return out


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResSeg(nn.Module): #HS-CNNR

    def __init__(self, input_channs, output_channs, stride=1, sr=False):
        super(ResSeg, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs
        self.sr = sr

        self.conv1 = conv3x3(self.input_channs, 64, stride)
        self.conv2 = ResBlock(64, 64, stride)
        self.conv3 = ResBlock(64, 64, stride)
        self.conv4 = ResBlock(64, 64, stride)
        self.conv5 = ResBlock(64, 64, stride)
        self.conv6 = ResBlock(64, 64, stride)
        self.conv7 = ResBlock(64, 64, stride)
        self.conv8 = ResBlock(64, 64, stride)
        self.conv9 = ResBlock(64, 64, stride)
        self.conv10 = ResBlock(64, 64, stride)
        self.conv11 = ResBlock(64, 64, stride)
        self.conv12 = ResBlock(64, 64, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = conv1x1(64, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the divide  2./n  carefully

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = x + identity
        x = self.relu(x)
        x = self.conv_out(x)

        return x


class ResBlockBN(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlockBN, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResSegBN(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(ResSegBN, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride)
        self.conv2 = ResBlockBN(64, 128, stride)
        self.conv3 = ResBlockBN(128, 256, stride)
        self.conv4 = ResBlockBN(256, 512, stride)
        self.conv5 = ResBlockBN(512, 256, stride)
        self.conv6 = ResBlockBN(256, 128, stride)
        self.conv7 = ResBlockBN(128, 64, stride)
        self.conv8 = ResBlockBN(64, 32, stride)
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv_out(x)

        return x


class ResBlockBN_pool(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlockBN_pool, self).__init__()

        self.conv_bottleneck = conv1x1(inplanes, planes, stride)
        self.bottleneck_norm = nn.BatchNorm2d(planes)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = self.maxpool(x)

        identity = self.conv_bottleneck(x)
        identity = self.bottleneck_norm(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResBlock_pool(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock_pool, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):

        x = self.maxpool(x)

        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResSegUBN_t(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(ResSegUBN_t, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride) # 1 / 1
        self.conv2 = ResBlockBN_pool(64, 128, stride) # 1 / 2
        self.conv3 = ResBlockBN_pool(128, 256, stride) # 1 / 4
        self.conv4 = ResBlockBN_pool(256, 512, stride) # 1 / 8
        self.conv5 = ResBlockBN_pool(512, 512, stride)  # 1 / 16
        self.conv1_u = BlockUpT(512, 256, stride) # 1 / 8
        self.conv2_u = BlockUpT(256, 128, stride)  # 1 / 4
        self.conv3_u = BlockUpT(128, 64, stride)  # 1 / 2
        self.conv4_u = BlockUpT(64, 32, stride)  # 1 / 1
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.conv1_u(x5, x4)
        x = self.conv2_u(x, x3)
        x = self.conv3_u(x, x2)
        x = self.conv4_u(x, x1)

        x = self.conv_out(x)

        return x


class ResSegBNU(nn.Module):

    def __init__(self, input_channs, output_channs, stride=1):
        super(ResSegBNU, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs

        self.conv1 = conv3x3(self.input_channs, 64, stride) # 1 / 1
        self.conv2 = ResBlockBN_pool(64, 128, stride) # 1 / 2
        self.conv3 = ResBlockBN_pool(128, 256, stride) # 1 / 4
        self.conv4 = ResBlockBN_pool(256, 512, stride) # 1 / 8
        self.conv5 = ResBlockBN_pool(256, 512, stride)  # 1 / 16
        self.conv1_u = BlockUp(512, 256, stride) # 1 / 8
        self.conv2_u = BlockUp(256, 128, stride)  # 1 / 4
        self.conv3_u = BlockUp(128, 64, stride)  # 1 / 2
        self.conv4_u = BlockUp(64, 32, stride)  # 1 / 1
        self.conv_out = conv1x1(32, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the devide  2./n  carefully

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.conv1_u(x5, x4)
        x = self.conv2_u(x, x3)
        x = self.conv3_u(x, x2)
        x = self.conv4_u(x, x1)

        x = self.conv_out(x)

        return x


class ResSegU(nn.Module): # HS-Unet-R

    def __init__(self, input_channs, output_channs, stride=1, sr=False):
        super(ResSegU, self).__init__()

        self.input_channs = input_channs
        self.output_channs = output_channs
        self.sr = sr

        self.conv1 = conv3x3(self.input_channs, 64, stride) # 1 / 1
        self.conv2 = ResBlock_pool(64, 64, stride) # 1 / 2
        self.conv3 = ResBlock_pool(64, 64, stride) # 1 / 4
        self.conv4 = ResBlock_pool(64, 64, stride) # 1 / 8
        self.conv5 = ResBlock_pool(64, 64, stride)  # 1 / 16
        self.conv1_u = BlockUpT(128, 64, stride) # 1 / 8
        self.conv2_u = BlockUpT(128, 64, stride)  # 1 / 4
        self.conv3_u = BlockUpT(128, 64, stride)  # 1 / 2
        self.conv4_u = BlockUpT(128, 64, stride)  # 1 / 1
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = conv1x1(64, self.output_channs, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))  # the divide  2./n  carefully

    def forward(self, x):

        x1 = self.conv1(x)

        [_, _, h, w] = x1.size() # for final crop and cat op

        identity = x1
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x = self.conv1_u(x5, x4)
        x = self.conv2_u(x, x3)
        x = self.conv3_u(x, x2)
        x = self.conv4_u(x, x1)

        x = x + identity
        x = self.relu(x)

        x = self.conv_out(x)

        return x
