#coding: 'utf-8'

"""
Hu2018
test_with_viisualization

created by Kazunari on 2018/10/04 
"""

import os
import os.path as osp
import time

import matplotlib
import matplotlib.image
import numpy as np
import torch
import torch.nn.parallel
from torch.nn.functional import relu
from tqdm import tqdm

from dataset import loaddata
from models import modules, net, resnet, densenet, senet
from utils import sobel, util

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}

nyud_root_path = "/home/takagi.kazunari/projects/datasets/hu_nyud"
sun_root_path = "/home/takagi.kazunari/projects/datasets/SUN_Depth"

def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True, pcamff=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('/home/takagi.kazunari/projects/TBDP-Net/runs/1231_TSM_PCAMFFv2_b16_step3/checkpoints/checkpoint_16.path.tar'))

    out_dir = "/home/takagi.kazunari/projects/TBDP-Net/experimental_results/output_npys/1231_tsm_pcamff_v2_e16"

    #makedirs_for_outputs(out_dir)

    test_loader = loaddata.getTestingData(nyud_root_path, 1)
    visualize_error_map(test_loader, model, out_dir)


def visualize_error_map(test_loader, model, out_dir):
    model.eval()

    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(test_loader)):
            f_num = str(i + 1).zfill(5)

            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda(async=True)
            image = image.cuda()

            image = torch.autograd.Variable(image, volatile=True)
            depth = torch.autograd.Variable(depth, volatile=True)

            #out_mff, out_semff = model(image)
            start = time.time()
            out_mff, out_semff = model(image)
            end = time.time()
            time_elapsed = end - start
            print(time_elapsed)

            out_mff = torch.nn.functional.upsample(out_mff, size=[depth.size(2), depth.size(3)], mode='bilinear')
            out_semff = torch.nn.functional.upsample(out_semff, size=[depth.size(2), depth.size(3)], mode='bilinear')

            os.makedirs(osp.join(out_dir, "mff", "npys"), exist_ok=True)
            np.save(osp.join(out_dir, "mff", "npys", "%s.npy" % f_num), out_mff.data.cpu().numpy())
            os.makedirs(osp.join(out_dir, "pcamff", "npys"), exist_ok=True)
            np.save(osp.join(out_dir, "pcamff", "npys", "%s.npy" % f_num), out_semff.data.cpu().numpy())


def define_model(is_resnet, is_densenet, is_senet, model='tbdp', parallel=False, semff=False, pcamff=False):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        if model == 'tbdp':
            model = net.TBDPNet(Encoder, num_features=2048,
                                 block_channel=[256, 512, 1024, 2048],
                                 parallel=parallel, pcamff=pcamff)
        elif model == 'hu':
            model = net.Hu(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], semff=semff, pcamff=pcamff)
        else:
            raise NotImplementedError("Select model type in [\'tbdp\', \'hu\']")
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        if model == 'tbdp':
            model = net.TBDPNet(Encoder, num_features=2208,
                                 block_channel=[192, 384, 1056, 2208],
                                 parallel=parallel, pcamff=pcamff)
        elif model == 'hu':
            model = net.Hu(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208], semff=semff, pcamff=pcamff)
        else:
            raise NotImplementedError("Select model type in [\'tbdp\', \'hu\']")
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        if model == 'tbdp':
            model = net.TBDPNet(Encoder, num_features=2048,
                                 block_channel=[256, 512, 1024, 2048],
                                 parallel=parallel, pcamff=pcamff)
        elif model == 'hu':
            model = net.Hu(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], semff=semff, pcamff=pcamff)
        else:
            raise NotImplementedError("Select model type in [\'tbdp\', \'hu\']")

    return model

def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
                 torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel

def makedirs_for_outputs(out_dir):
    if not osp.exists(osp.join(out_dir, "images")):
        os.makedirs(osp.join(out_dir, "images"))
    if not osp.exists(osp.join(out_dir, "depths")):
        os.makedirs(osp.join(out_dir, "depths"))
    if not osp.exists(osp.join(out_dir, "npys")):
        os.makedirs(osp.join(out_dir, "npys"))
    if not osp.exists(osp.join(out_dir, "refinement", "depths")):
        os.makedirs(osp.join(out_dir, "refinement", "depths"))
    if not osp.exists(osp.join(out_dir, "refinement", "error_maps")):
        os.makedirs(osp.join(out_dir, "refinement", "error_maps"))
    if not osp.exists(osp.join(out_dir, "refinement", "delta_1")):
        os.makedirs(osp.join(out_dir, "refinement", "delta_1"))
    if not osp.exists(osp.join(out_dir, "refinement", "rel")):
        os.makedirs(osp.join(out_dir, "refinement", "rel"))
    if not osp.exists(osp.join(out_dir, "base", "depths")):
        os.makedirs(osp.join(out_dir, "base", "depths"))
    if not osp.exists(osp.join(out_dir, "base", "error_maps")):
        os.makedirs(osp.join(out_dir, "base", "error_maps"))


if __name__ == '__main__':
    main()
