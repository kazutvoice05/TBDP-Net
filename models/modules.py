from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np

from . import senet
from . import resnet
from utils.visualize_util import visualize_array

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class E_resnet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_resnet, self).__init__()        
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class E_densenet(nn.Module):

    def __init__(self, original_model, num_features = 2208):
        super(E_densenet, self).__init__()        
        self.features = original_model.features

    def forward(self, x):
        x01 = self.features[0](x)
        x02 = self.features[1](x01)
        x03 = self.features[2](x02)
        x04 = self.features[3](x03)

        x_block1 = self.features[4](x04)
        x_block1 = self.features[5][0](x_block1)
        x_block1 = self.features[5][1](x_block1)
        x_block1 = self.features[5][2](x_block1)
        x_tran1 = self.features[5][3](x_block1)

        x_block2 = self.features[6](x_tran1)
        x_block2 = self.features[7][0](x_block2)
        x_block2 = self.features[7][1](x_block2)
        x_block2 = self.features[7][2](x_block2)
        x_tran2 = self.features[7][3](x_block2)

        x_block3 = self.features[8](x_tran2)
        x_block3 = self.features[9][0](x_block3)
        x_block3 = self.features[9][1](x_block3)
        x_block3 = self.features[9][2](x_block3)
        x_tran3 = self.features[9][3](x_block3)

        x_block4 = self.features[10](x_tran3)
        x_block4 = F.relu(self.features[11](x_block4))

        return x_block1, x_block2, x_block3, x_block4

class E_senet(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(E_senet, self).__init__()        
        self.base = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.base[0](x)
        x_block1 = self.base[1](x)
        x_block2 = self.base[2](x_block1)
        x_block3 = self.base[3](x_block2)
        x_block4 = self.base[4](x_block3)

        return x_block1, x_block2, x_block3, x_block4

class D(nn.Module):

    def __init__(self, num_features = 2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features //
                               2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x = F.relu(self.bn(self.conv(x_block4)))
        x = self.up1(x, [x_block3.size(2), x_block3.size(3)])
        x = self.up2(x, [x_block2.size(2), x_block2.size(3)])
        x = self.up3(x, [x_block1.size(2), x_block1.size(3)])
        x = self.up4(x, [x_block1.size(2)*2, x_block1.size(3)*2])

        return x

class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64, save=False):

        super(MFF, self).__init__()
        
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)
        
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)
       
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)
       
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

        self.save = save
        self.f_num = 1

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class SEMFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(SEMFF, self).__init__()

        self.se_reduction = 16

        self.se1 = senet.SEModule(block_channel[0], self.se_reduction)
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.se2 = senet.SEModule(block_channel[1], self.se_reduction)
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.se3 = senet.SEModule(block_channel[2], self.se_reduction)
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.se4 = senet.SEModule(block_channel[3], self.se_reduction)
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

        self.f_num = 1
        print("SEMFF is used as SEMFF module.")

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(self.se1(x_block1), size)
        x_m2 = self.up2(self.se2(x_block2), size)
        x_m3 = self.up3(self.se3(x_block3), size)
        x_m4 = self.up4(self.se4(x_block4), size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        """ SAVE VISUALIZES FEATURE MAPS """
        """
        dir_path = "/home/takagi.kazunari/projects/TBDP-Net/experimental_results/feature_map_visualization/TSM_1214/"
        visualize_array(x_m1.mean(1).data.cpu().numpy(), cmap="hot",
                        f_name=dir_path + "block_1/%s.png" % str(self.f_num).zfill(5))
        visualize_array(x_m2.mean(1).data.cpu().numpy(), cmap="hot",
                        f_name=dir_path + "block_2/%s.png" % str(self.f_num).zfill(5))
        visualize_array(x_m3.mean(1).data.cpu().numpy(), cmap="hot",
                        f_name=dir_path + "block_3/%s.png" % str(self.f_num).zfill(5))
        visualize_array(x_m4.mean(1).data.cpu().numpy(), cmap="hot",
                        f_name=dir_path + "block_4/%s.png" % str(self.f_num).zfill(5))
        visualize_array(x.mean(1).data.cpu().numpy(), cmap="hot",
                        f_name=dir_path + "out/%s.png" % str(self.f_num).zfill(5))

        self.f_num += 1
        """

        return x

# PCAMFF V1
"""
class PCAMFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(PCAMFF, self).__init__()

        print("PCAMFF V1 is used")

        self.se_reduction = 16

        self.pca1 = PCA(block_channel[0], 36, self.se_reduction)
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.pca2 = PCA(block_channel[1], 72, self.se_reduction)
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.pca3 = PCA(block_channel[2], 128, self.se_reduction)
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.pca4 = PCA(block_channel[3], 0, self.se_reduction)
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

        self.f_num = 1
        print("PCAMFF is used as SEMFF module.")

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m4, r_4 = self.pca4(x_block4, None)
        x_m3, r_3 = self.pca3(x_block3, r_4)
        x_m2, r_2 = self.pca2(x_block2, r_3)
        x_m1, r_1 = self.pca1(x_block1, r_2)

        x_m1 = self.up1(x_m1, size)
        x_m2 = self.up2(x_m2, size)
        x_m3 = self.up3(x_m3, size)
        x_m4 = self.up4(x_m4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x
"""


# PCAMFF V2 ~
class PCAMFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(PCAMFF, self).__init__()

        self.se_reduction = 16

        self.pca1 = PCA(block_channel[0], block_channel[1], self.se_reduction)
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.pca2 = PCA(block_channel[1], block_channel[2], self.se_reduction)
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.pca3 = PCA(block_channel[2], block_channel[3], self.se_reduction)
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.pca4 = PCA(block_channel[3], 0, self.se_reduction)
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

        self.f_num = 1
        print("PCAMFF is used as SEMFF module.")

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m4 = self.pca4(x_block4, None)
        x_m3 = self.pca3(x_block3, x_m4)
        x_m2 = self.pca2(x_block2, x_m3)
        x_m1 = self.pca1(x_block1, x_m2)

        x_m1 = self.up1(x_m1, size)
        x_m2 = self.up2(x_m2, size)
        x_m3 = self.up3(x_m3, size)
        x_m4 = self.up4(x_m4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class CSMFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(CSMFF, self).__init__()

        print("CSMFF is used as SEMFF module.")

        self.ca_reduction = 16

        self.ca1 = ChannelAttention(block_channel[0], self.ca_reduction)
        self.sa1 = SpatialAttention()
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.ca2 = ChannelAttention(block_channel[1], self.ca_reduction)
        self.sa2 = SpatialAttention()
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.ca3 = ChannelAttention(block_channel[2], self.ca_reduction)
        self.sa3 = SpatialAttention()
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.ca4 = ChannelAttention(block_channel[3], self.ca_reduction)
        self.sa4 = SpatialAttention()
        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        #self.ca_out = ChannelAttention(num_features, self.ca_reduction)
        #self.sa_out = SpatialAttention()
        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(self.sa1(self.ca1(x_block1)), size)
        x_m2 = self.up2(self.sa2(self.ca2(x_block2)), size)
        x_m3 = self.up3(self.sa3(self.ca3(x_block3)), size)
        x_m4 = self.up4(self.sa4(self.ca4(x_block4)), size)

        x = torch.cat((x_m1, x_m2, x_m3, x_m4), 1)
        x = self.bn(self.conv(x))
        x = F.relu(x)

        return x

class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()

        self.roi_align = RoIAlign(None, None)

    def forward(self, x_block1, x_block2, x_block3, x_block4, x_decoder, bbox):
        """

        :param x_block1:
        :param x_block2:
        :param x_block3:
        :param x_block4:
        :param bbox: original Bounding Box [x1, y1, x2, y2]
        :return:
        """

        box_idx = torch.Tensor([0]).type(torch.int).cuda()

        bbox_1 = torch.div(bbox, 4)
        h = torch.ceil(bbox_1[0, 3] - bbox_1[0, 1] + 1 / 4).data.cpu().numpy()
        w = torch.ceil(bbox_1[0, 2] - bbox_1[0, 0] + 1 / 4).data.cpu().numpy()
        cx_block1 = self.roi_align(x_block1, bbox_1, box_idx, crop_size=(h, w))

        bbox_2 = torch.div(bbox, 8)
        h = torch.ceil(bbox_2[0, 3] - bbox_2[0, 1] + 1 / 8).data.cpu().numpy()
        w = torch.ceil(bbox_2[0, 2] - bbox_2[0, 0] + 1 / 8).data.cpu().numpy()
        cx_block2 = self.roi_align(x_block2, bbox_2, box_idx, crop_size=(h, w))

        bbox_3 = torch.div(bbox, 16)
        h = torch.ceil(bbox_3[0, 3] - bbox_3[0, 1] + 1 / 16).data.cpu().numpy()
        w = torch.ceil(bbox_3[0, 2] - bbox_3[0, 0] + 1 / 16).data.cpu().numpy()
        cx_block3 = self.roi_align(x_block3, bbox_3, box_idx, crop_size=(h, w))

        bbox_4 = torch.div(bbox, 32)
        h = torch.ceil(bbox_4[0, 3] - bbox_4[0, 1] + 1 / 32).data.cpu().numpy()
        w = torch.ceil(bbox_4[0, 2] - bbox_4[0, 0] + 1 / 32).data.cpu().numpy()
        cx_block4 = self.roi_align(x_block4, bbox_4, box_idx, crop_size=(h, w))

        bbox_d = torch.div(bbox, 2)
        h = torch.ceil(bbox_d[0, 3] - bbox_d[0, 1] + 1 / 2).data.cpu().numpy()
        w = torch.ceil(bbox_d[0, 2] - bbox_d[0, 0] + 1 / 2).data.cpu().numpy()
        cx_decoder = self.roi_align(x_decoder, bbox_d, box_idx, crop_size=(h, w))

        return cx_block1, cx_block2, cx_block3, cx_block4, cx_decoder


class R(nn.Module):
    def __init__(self, block_channel):

        super(R, self).__init__()
        
        num_features = 64 + block_channel[3]//32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)

        return x

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.f_conv = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.f_bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.f_bn(self.f_conv(x)))

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


""" PCA V1"""
class PCA(nn.Module):
    def __init__(self, channels, add_channels, reduction):
        super(PCA, self).__init__()

        print("PCA V1 is used.")

        summed_channels = channels + add_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(summed_channels, summed_channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(summed_channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        module_input = x
        x = self.avg_pool(x)

        if y is not None:
            y = self.avg_pool(y)
            x = torch.cat([x, y], dim=1)

        x_1 = self.fc1(x)
        x_1 = self.relu(x_1)
        x_2 = self.fc2(x_1)
        x_2 = self.sigmoid(x_2)
        return module_input * x_2, x_1


""" PCA V2"""
class PCA(nn.Module):
    def __init__(self, channels, add_channels, reduction):
        super(PCA, self).__init__()

        summed_channels = channels + add_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(summed_channels, summed_channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(summed_channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        module_input = x
        x = self.avg_pool(x)

        if y is not None:
            y = self.avg_pool(y)
            x = torch.cat([x, y], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

""" PCA v4"""
"""
class PCA(nn.Module):
    def __init__(self, channels, add_channels, reduction):
        super(PCA, self).__init__()

        print("PCA V4")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        summed_channels = channels + add_channels // reduction
        if add_channels != 0:
            self.fc1_2 = nn.Conv2d(add_channels, add_channels // reduction, kernel_size=1,
                                   padding=0)

        self.fc1_1 = nn.Conv2d(summed_channels, summed_channels // reduction, kernel_size=1,
                             padding=0)
        self.fc2 = nn.Conv2d(summed_channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        module_input = x
        x = self.avg_pool(x)

        if y is not None:
            y = self.avg_pool(y)
            y = self.relu(self.fc1_2(y))
            x = torch.cat([x, y], dim=1)

        x = self.fc1_1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
"""

""" Cascaded Channel Attention Module """
class CCA(nn.Module):
    def __init__(self, channels, add_channels, reduction):
        super(CCA, self).__init__()

        summed_channels = channels + add_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(summed_channels, summed_channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(summed_channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        module_input = x

        x_avg = torch.cat([self.avg_pool(x), self.avg_pool(y)], dim=1)
        x_max = torch.cat([self.max_pool(x), self.max_pool(y)], dim=1)

        x_avg = self.fc2(self.relu(self.fc1(x_avg)))
        x_max = self.fc2(self.relu(self.fc1(x_max)))

        x = self.sigmoid(x_avg + x_max)

        return module_input * x

""" Cascaded Spatial Attention"""
class CSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(CSA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        y_avg = F.upsample(torch.mean(y, dim=1, keepdim=True), size=[x.size(2), x.size(3)], mode='bilinear')
        y_max, _ = torch.max(y, dim=1, keepdim=True)
        y_max = F.upsample(y_max, size=[x.size(2), x.size(3)], mode='bilinear')

        out = torch.cat([x_avg, x_max, y_avg, y_max], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


""" Cascaded Channel-Spatial Attention Module """
class CCSAMFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(CCSAMFF, self).__init__()

        self.se_reduction = 16

        self.cca1 = CCA(block_channel[0], block_channel[1], self.se_reduction)
        self.csa1 = CSA()
        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16)

        self.cca2 = CCA(block_channel[1], block_channel[2], self.se_reduction)
        self.csa2 = CSA()
        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16)

        self.cca3 = CCA(block_channel[2], block_channel[3], self.se_reduction)
        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16)

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16)

        self.conv = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

        self.f_num = 1
        print("CCSAMFF is used as SEMFF module.")

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        # Channel Attention Phase
        x_m3 = self.cca3(x_block3, x_block4)
        x_m2 = self.cca2(x_block2, x_m3)
        x_m1 = self.cca1(x_block1, x_m2)

        # Spatial Attention Phase
        x_m2 = self.csa2(x_m2, x_m3)
        x_m1 = self.csa1(x_m1, x_m2)

        # Multi-scale Feature Fusion Phase
        x_m1 = self.up1(x_m1, size)
        x_m2 = self.up2(x_m2, size)
        x_m3 = self.up3(x_m3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x
