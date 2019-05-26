#coding: 'utf-8'

"""
dinet_2
visualize_util

created by Kazunari on 2018/10/22 
"""

import argparse
import os.path as osp
import os
import sys
import numpy as np
import cv2
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.figure
import matplotlib

import torch

SUNRGBD_MEAN = [0.49377292, 0.45705342, 0.4330905]
SUNRGBD_STD = [0.25613034, 0.26139295, 0.26440033]

NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]

def visualize_array(src, cmap="gray", f_name=None):
    """

    :param src: image source array [c, h, w]
    :param cmap:
    :return: depth image visualized by selected color map (matplotlib.image object)
    """

    if isinstance(src, np.ndarray):
        src = src
    elif isinstance(src, torch.Tensor):
        src = src.data.numpy()
    else:
        raise NotImplementedError()

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(dpi=100, frameon=False)
    canvas = FigureCanvas(fig)

    if src.shape[0] == 1:
        im = fig.figimage(src[0], cmap=cmap, resize=True)
    else:
        im = fig.figimage(src.transpose(1, 2, 0), resize=True)

    if f_name is not None:
        fig.savefig(f_name, dpi=100, transparent=True)
        return fig
    else:
        return fig

def denormalize_image(image, mode = "sun"):
    if isinstance(image, np.ndarray):
        if image.shape[0] != 3:
            image = image.transpose((2, 0, 1))
        if mode == "sun":
            image[0, :, :] = (image[0, :, :] * SUNRGBD_STD[0]) + SUNRGBD_MEAN[0]
            image[1, :, :] = (image[1, :, :] * SUNRGBD_STD[1]) + SUNRGBD_MEAN[1]
            image[2, :, :] = (image[2, :, :] * SUNRGBD_STD[2]) + SUNRGBD_MEAN[2]
        elif mode == "nyud":
            image[0, :, :] = (image[0, :, :] * NYUD_STD[0]) + NYUD_MEAN[0]
            image[1, :, :] = (image[1, :, :] * NYUD_STD[1]) + NYUD_MEAN[1]
            image[2, :, :] = (image[2, :, :] * NYUD_STD[2]) + NYUD_MEAN[2]

        image = image * 255

        return image
    else:
        raise NotImplementedError("input image array must be numpy.ndarray.")