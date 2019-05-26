#coding: 'utf-8'

"""
Hu2018
test_with_viisualization

created by Kazunari on 2018/10/04 
"""

import os
import os.path as osp

import matplotlib
import matplotlib.image
import numpy as np
import torch
import torch.nn.parallel
from torch.nn.functional import relu
from tqdm import tqdm
import glob

from dataset import loaddata
from models import modules, net, resnet, densenet, senet
from utils import sobel, util

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}

def main():
    #model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    #model = torch.nn.DataParallel(model).cuda()
    #model.load_state_dict(torch.load('/home/takagi.kazunari/projects/LDPNet/runs/1116_SENet_SEMFF_NYUD/checkpoints/checkpoint_8.path.tar'))

    out_dir = "/home/takagi.kazunari/projects/TBDP-Net/hu_tsm12_outputs"
    dataset_dir = "/home/takagi.kazunari/projects/datasets/hu_nyud"

    test_loader = loaddata.getTestingData(dataset_dir, 1)
    visualize_error_map(test_loader, out_dir)


def visualize_error_map(test_loader, out_dir):
    #model.eval()

    hu_paths = sorted(glob.glob("/home/takagi.kazunari/projects/TBDP-Net/experimental_results/output_npys/hu_1116/*"))
    tsm_paths = sorted(glob.glob("/home/takagi.kazunari/projects/TBDP-Net/experimental_results/output_npys/1231_tsm_pcamff_v2_e16/mff/*"))
    
    mff_out_dir = "/home/takagi.kazunari/projects/TBDP-Net/hu_vs_tsm_1231_winter/hu"
    tsm_out_dir = "/home/takagi.kazunari/projects/TBDP-Net/hu_vs_tsm_1231_winter/tsm"

    makedirs_for_outputs(mff_out_dir)
    makedirs_for_outputs(tsm_out_dir)

    errors = None

    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(test_loader)):
            f_num = str(i + 1).zfill(5)

            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda(async=True)
            image = image.cuda()

            image = torch.autograd.Variable(image, volatile=True)
            depth = torch.autograd.Variable(depth, volatile=True)

            #out_r = model(image)
            #out_r = torch.nn.functional.upsample(out_r, size=[depth.size(2), depth.size(3)], mode='bilinear')

            mff_out = torch.Tensor(np.load(hu_paths[i])).cuda()
            tsm_out = torch.Tensor(np.load(tsm_paths[i])).cuda()

            _, _target, Mask, ValidElement = util.setZeroToNum(mff_out, depth)

            mff_error = torch.abs(mff_out - depth)
            tsm_error = torch.abs(tsm_out - depth)

            mff_error = mff_error * Mask.float()
            tsm_error = tsm_error * Mask.float()

            mff_rms = np.sqrt(mff_error.sum().data.cpu().numpy() / ValidElement)
            tsm_rms = np.sqrt(tsm_error.sum().data.cpu().numpy() / ValidElement)

            if errors is None:
                errors = np.array([[mff_rms, tsm_rms]])
            else:
                errors = np.concatenate((errors, [[mff_rms, tsm_rms]]))

            mff_delta = torch.max(depth / mff_out, mff_out / depth)
            tsm_delta = torch.max(depth / tsm_out, tsm_out / depth)
            mff_delta = torch.where(mff_delta < 1.25, torch.zeros(mff_delta.shape).cuda(), torch.ones(mff_delta.shape).cuda())
            tsm_delta = torch.where(tsm_delta < 1.25, torch.zeros(tsm_delta.shape).cuda(), torch.ones(tsm_delta.shape).cuda())

            image = image.data.cpu().numpy()[0]

            mean = np.expand_dims(np.expand_dims(imagenet_stats['mean'], axis=1), axis=1)
            std = np.expand_dims(np.expand_dims(imagenet_stats['std'], axis=1), axis=1)

            image = np.clip(np.floor((image * std + mean) * 255), 0, 255).transpose(1, 2, 0)


            # save image
            matplotlib.image.imsave(osp.join(mff_out_dir, "images", "%s.png" % f_num),
                                    np.array(image, dtype=np.uint8))

            d_vmin = min(depth.min(), mff_out.min(), tsm_out.min())
            d_vmax = max(depth.max(), mff_out.max(), tsm_out.max())

            # save depth
            matplotlib.image.imsave(osp.join(mff_out_dir, "depths", "%s.png" % f_num),
                                    depth.view(depth.size(2), depth.size(3)).data.cpu().numpy(),
                                    vmin=d_vmin, vmax=d_vmax, cmap='winter')


            # save refinement's outputs
            matplotlib.image.imsave(osp.join(mff_out_dir, "preds", "%s.png" % f_num),
                                    mff_out.view(mff_out.size(2), mff_out.size(3)).data.cpu().numpy(),
                                    vmin=d_vmin, vmax=d_vmax, cmap='winter')
            df_name = osp.join(tsm_out_dir, "preds", "%s.png" % f_num)
            save_fig_with_colorbar(tsm_out.data.cpu().numpy()[0, 0], d_vmin, d_vmax, f_name=df_name)
            #matplotlib.image.imsave(osp.join(tsm_out_dir, "preds", "%s.png" % f_num),
            #                        tsm_out.view(tsm_out.size(2), tsm_out.size(3)).data.cpu().numpy(),
            #                        vmin=d_vmin, vmax=d_vmax, cmap='winter')



            e_vmin = min(mff_error.min(), tsm_error.min())
            e_vmax = max(mff_error.max(), tsm_error.max())

            # save error maps
            matplotlib.image.imsave(osp.join(mff_out_dir, "error_maps", "%s.png" % f_num),
                                    mff_error.view(mff_error.size(2), mff_error.size(3)).data.cpu().numpy(),
                                    vmin=e_vmin, vmax=e_vmax, cmap='hot')

            ef_name = osp.join(tsm_out_dir, "error_maps", "%s.png" % f_num)
            save_fig_with_colorbar(tsm_error.data.cpu().numpy()[0, 0], e_vmin, e_vmax, f_name=ef_name, cmap="hot")
            #matplotlib.image.imsave(osp.join(tsm_out_dir, "error_maps", "%s.png" % f_num),
            #                        tsm_error.view(tsm_error.size(2), tsm_error.size(3)).data.cpu().numpy(),
            #                        vmin=e_vmin, vmax=e_vmax, cmap='hot')

            """
            try:
                matplotlib.image.imsave(osp.join(mff_out_dir, "refinement", "delta_1", "%s.png" % f_num),
                                        mff_delta.view(mff_delta.size(2), mff_delta.size(3)).data.cpu().numpy(),
                                        vmin=1.25, vmax=max(mff_delta.max(), semff_delta.max()), cmap='hot')
                matplotlib.image.imsave(osp.join(semff_out_dir, "refinement", "delta_1", "%s.png" % f_num),
                                        semff_delta.view(semff_delta.size(2), semff_delta.size(3)).data.cpu().numpy(),
                                        vmin=1.25, vmax=max(mff_delta.max(), semff_delta.max()), cmap='hot')
            except:
                matplotlib.image.imsave(osp.join(mff_out_dir, "refinement", "delta_1", "%s.png" % f_num),
                                        mff_delta.view(mff_delta.size(2), mff_delta.size(3)).data.cpu().numpy(),
                                        vmin=min(mff_delta.min(), semff_delta.min()),
                                        vmax=max(mff_delta.max(), semff_delta.max()), cmap='hot')
                matplotlib.image.imsave(osp.join(semff_out_dir, "refinement", "delta_1", "%s.png" % f_num),
                                        semff_delta.view(semff_delta.size(2), semff_delta.size(3)).data.cpu().numpy(),
                                        vmin=min(mff_delta.min(), semff_delta.min()),
                                        vmax=max(mff_delta.max(), semff_delta.max()), cmap='hot')
            """

            matplotlib.image.imsave(osp.join(mff_out_dir, "delta_1", "%s.png" % f_num),
                                    mff_delta.view(mff_delta.size(2), mff_delta.size(3)).data.cpu().numpy(),
                                    cmap='gray')
            matplotlib.image.imsave(osp.join(tsm_out_dir, "delta_1", "%s.png" % f_num),
                                    tsm_delta.view(tsm_delta.size(2), tsm_delta.size(3)).data.cpu().numpy(),
                                    cmap='gray')


            """
            matplotlib.image.imsave(osp.join(mff_out_dir, "refinement", "rel", "%s.png" % f_num),
                                    mff_rel.view(mff_rel.size(2), mff_rel.size(3)).data.cpu().numpy(),
                                    vmin=0.0, vmax=max(mff_rel.max(), semff_rel.max()), cmap='hot')
            matplotlib.image.imsave(osp.join(semff_out_dir, "refinement", "rel", "%s.png" % f_num),
                                    semff_rel.view(semff_rel.size(2), semff_rel.size(3)).data.cpu().numpy(),
                                    vmin=0.0, vmax=max(mff_rel.max(), semff_rel.max()), cmap='hot')
            """


            #np.save(osp.join(out_dir, "npys", "%s.npy" % f_num), out_r.data.cpu().numpy())

        np.savetxt("1231_tsm_mff_pcamff.csv", errors, delimiter=",")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

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
    if not osp.exists(osp.join(out_dir, "preds")):
        os.makedirs(osp.join(out_dir, "preds"))
    if not osp.exists(osp.join(out_dir, "error_maps")):
        os.makedirs(osp.join(out_dir, "error_maps"))
    if not osp.exists(osp.join(out_dir, "delta_1")):
        os.makedirs(osp.join(out_dir, "delta_1"))
    if not osp.exists(osp.join(out_dir, "rel")):
        os.makedirs(osp.join(out_dir, "rel"))


def save_fig_with_colorbar(array, vmin, vmax, f_name="tmp", cmap="winter"):

    fig = plt.figure()
    plt.tight_layout()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(array, cmap=cmap, vmax=vmax, vmin=vmin)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.05)

    # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
    # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

    diff = vmax - vmin

    ticks = [round(diff * 0.05 + vmin, 2),
             round(diff * 0.25 + vmin, 2),
             round(diff * 0.5 + vmin, 2),
             round(diff * 0.75 + vmin, 2),
             round(diff * 0.95 + vmin, 2)]

    cbar = fig.colorbar(im, cax=cax, ticks=ticks, cmap=cmap)
    cbar.ax.set_xticklabels(ticks)

    fig.savefig("%s.png" % f_name)


if __name__ == '__main__':
    main()