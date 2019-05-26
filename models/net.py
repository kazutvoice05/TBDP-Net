from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from . import modules
from torchvision import utils
from utils.visualize_util import visualize_array

from . import senet
from . import resnet
from . import densenet


class Hu(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, semff=False, pcamff=False):
        print("Hu model is used")

        super(Hu, self).__init__()

        self.semff = semff

        self.E = Encoder
        self.D = modules.D(num_features)

        if semff:
            self.MFF = modules.SEMFF(block_channel)
            print("SEMFF is used as MFF Module.")
        elif pcamff:
            self.MFF = modules.PCAMFF(block_channel)
            print("PCAMFF is used as MFF Module.")
        else:
            self.MFF = modules.MFF(block_channel)
            print("MFF is used as MFF Module.")
        self.R = modules.R(block_channel)

        self.f_num = 0

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out

    def test_forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])

        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out, x_decoder


class TBDPNet(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, parallel=False, pretrained_model=None, pcamff=False):
        print("TBDP-Net is used.")
        super(TBDPNet, self).__init__()

        self.parallel = parallel

        self.E = Encoder
        self.D = modules.D(num_features)

        self.MFF = modules.MFF(block_channel)

        if pcamff:
            #self.SEMFF = modules.PCAMFF(block_channel)
            self.SEMFF = modules.CCSAMFF(block_channel)
        else:
            self.SEMFF = modules.SEMFF(block_channel)

        if not parallel:
            print("Shared R Mode")
            self.R = modules.R(block_channel)
        else:
            print("Parallel R Mode")
            self.MFF_R = modules.R(block_channel)
            self.SEMFF_R = modules.R(block_channel)

        if pretrained_model is not None:
            self.load_pretrained_params(pretrained_model)


    def forward(self, image):
        x_block1, x_block2, x_block3, x_block4 = self.E(image)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        x_semff = self.SEMFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])

        if not self.parallel:
            mff_out = self.R(torch.cat((x_decoder, x_mff), 1))
            semff_out = self.R(torch.cat((x_decoder, x_semff), 1))
        else:
            mff_out = self.MFF_R(torch.cat((x_decoder, x_mff), 1))
            semff_out = self.SEMFF_R(torch.cat((x_decoder, x_semff), 1))

        return mff_out, semff_out

    def load_pretrained_params(self, model_path):
        pretrained_weights = torch.load(model_path)["state_dict"]

        e_weights = {}
        for key in pretrained_weights.keys():
            if "module.E." in key:
                n_key = key.replace("module.E.", "")
                e_weights[n_key] = pretrained_weights[key]

        self.E.load_state_dict(e_weights)

        d_weights = {}
        for key in pretrained_weights.keys():
            if "module.D." in key:
                n_key = key.replace("module.D.", "")
                d_weights[n_key] = pretrained_weights[key]

        self.D.load_state_dict(d_weights)

        mff_weights = {}
        for key in pretrained_weights.keys():
            if "module.MFF." in key:
                n_key = key.replace("module.MFF.", "")
                mff_weights[n_key] = pretrained_weights[key]

        self.MFF.load_state_dict(mff_weights)

        r_weights = {}
        for key in pretrained_weights.keys():
            if "module.R." in key:
                n_key = key.replace("module.R.", "")
                r_weights[n_key] = pretrained_weights[key]

        self.R.load_state_dict(r_weights)
