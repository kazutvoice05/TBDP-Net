import argparse
import os
import os.path as osp
import time

from dataset import loaddata, loaddata_sun
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from tensorboardX import SummaryWriter
from utils.visualize_util import *

import test
from models import modules, net, resnet, densenet, senet
from utils import sobel

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--step-size', default=5, type=int,
                    help='step_size of learning rate decreasing')
parser.add_argument('--start-epoch', default=1, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--name', '-n', default="tmp", type=str,
                    help='experiment name')
parser.add_argument('--model', default="tbdp", type=str,
                    help='use model type ["tbdp", "hu"]')
parser.add_argument('--semff', action="store_true",
                    help='Use SE_MFF Module')
parser.add_argument('--pcamff', action="store_true",
                    help='Use PCAMFF Module')
parser.add_argument('--parallel', action="store_true",
                    help='Use parallel R')
parser.add_argument('--dataset', default="nyud", type=str,
                    help='used dataset ["nyud", "sun"]')
parser.add_argument('--out_root', '-o', default="./runs", type=str,
                    help='root output directory')
parser.add_argument('--pretrained_model',
                    default=None,
                    type=str, help='root output directory')

nyud_root_path = "/home/takagi.kazunari/projects/datasets/hu_nyud"
sun_root_path = "/home/takagi.kazunari/projects/datasets/SUN_Depth"


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
   

def main():
    global args
    args = parser.parse_args()
    out_dir = os.path.join(args.out_root, args.name)

    writer = SummaryWriter(out_dir)

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True,
                         model=args.model, parallel=args.parallel, semff=args.semff, pcamff=args.pcamff)

    gpu_num = torch.cuda.device_count()
    batch_size_per_gpu = 4

    device_ids = []
    for i in range(gpu_num):
        device_ids.append(i)

    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    batch_size = gpu_num * batch_size_per_gpu

    cudnn.benchmark = True

    """ Set Different Learning Rate """
    """
    params_with_lr = []
    for name, param in model.named_parameters():
        if "SEMFF.se" in name:
            params_with_lr.append({"params": param, "lr": args.lr/10})
        else:
            params_with_lr.append({"params": param})

    optimizer = torch.optim.Adam(params_with_lr, args.lr, weight_decay=args.weight_decay)
    """

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    print("step_size is set to %d" % scheduler.step_size)

    if args.dataset == 'nyud':
        train_loader = loaddata.getTrainingData(nyud_root_path, batch_size)
        test_loader = loaddata.getTestingData(nyud_root_path, batch_size)
    elif args.dataset == 'sun':
        train_loader = loaddata_sun.getTrainingData(sun_root_path, batch_size)
        test_loader = loaddata_sun.getTestingData(sun_root_path, batch_size)
    else:
        raise NotImplementedError('Specify dataset in [\'nyud\', \'sun\']')

    vis_out_dir = osp.join(out_dir, "outputs")

    for epoch in range(args.epochs):
        scheduler.step()
        e = epoch + args.start_epoch
        lr = args.lr * (0.1 ** (epoch // args.step_size))
        train(train_loader, model, optimizer, e, writer)

        results = None
        images = None
        pred_depths = None
        if isinstance(model.module, net.TBDPNet):
            mff_results, semff_results, images, mff_depths, semff_depths =\
                test.test_tbdp(test_loader, model, dataset=args.dataset, returnValue=True, returnSamples=True,
                                     sample_idx=[0, 1, 2, 3, 4, 5])
            results = [mff_results, semff_results]
            pred_depths = [mff_depths, semff_depths]

        elif isinstance(model.module, net.Hu):
            results, images, pred_depths = test.test_hu(test_loader, model,
                                                         dataset=args.dataset, returnValue=True, returnSamples=True,
                                                         sample_idx=[0, 1, 2, 3, 4, 5])

        for i in range(len(images)):
            if epoch == 0:
                os.makedirs(osp.join(vis_out_dir, "images"), exist_ok=True)
                image = np.clip(denormalize_image(images[i].data.cpu().numpy(), mode='nyud').astype(np.uint8), 0, 254)
                image = visualize_array(image, f_name=osp.join(vis_out_dir, "images", str(i+1) + ".png"))
            else:
                image = np.clip(denormalize_image(images[i].data.cpu().numpy(), mode='nyud').astype(np.uint8), 0, 254)
                image = visualize_array(image)

            writer.add_figure("image/%d" % (i + 1), image, e)

            if isinstance(model.module, net.TBDPNet):
                mff_depths, semff_depths = pred_depths
                os.makedirs(osp.join(vis_out_dir, "mff_depths", str(i+1)), exist_ok=True)
                os.makedirs(osp.join(vis_out_dir, "semff_depths", str(i+1)), exist_ok=True)

                mff_depth = visualize_array(mff_depths[i].data.cpu().numpy(),
                                            f_name=osp.join(vis_out_dir, "mff_depths", str(i+1), "%d_%d.png" % (i + 1, e)))
                semff_depth = visualize_array(semff_depths[i].data.cpu().numpy(),
                                              f_name=osp.join(vis_out_dir, "semff_depths", str(i+1), "%d_%d.png" % (i + 1, e)))

                writer.add_figure("mff_prediction/%d" % (i + 1), mff_depth, e)
                writer.add_figure("semff_prediction/%d" % (i + 1), semff_depth, e)
            elif isinstance(model.module, net.Hu):
                os.makedirs(osp.join(vis_out_dir, "pred", str(i+1)), exist_ok=True)

                depth = visualize_array(pred_depths[i].data.cpu().numpy(),
                                        f_name=osp.join(vis_out_dir, "pred", str(i+1), "%d_%d.png" % (i + 1, e)))
                writer.add_figure("prediction/%d" % (i+1), depth, e)

        if isinstance(model.module, net.TBDPNet):
            mff_results, semff_results = results

            writer.add_scalar("mff/RMSE", mff_results["RMSE"], e)
            writer.add_scalar("mff/ABS_REL", mff_results["ABS_REL"], e)
            writer.add_scalar("mff/LG10", mff_results["LG10"], e)
            writer.add_scalar("mff/DELTA1", mff_results["DELTA1"], e)
            writer.add_scalar("mff/DELTA2", mff_results["DELTA2"], e)
            writer.add_scalar("mff/DELTA3", mff_results["DELTA3"], e)
            writer.add_scalar("mff/lr", lr, e)
            writer.add_scalar("semff/RMSE", semff_results["RMSE"], e)
            writer.add_scalar("semff/ABS_REL", semff_results["ABS_REL"], e)
            writer.add_scalar("semff/LG10", semff_results["LG10"], e)
            writer.add_scalar("semff/DELTA1", semff_results["DELTA1"], e)
            writer.add_scalar("semff/DELTA2", semff_results["DELTA2"], e)
            writer.add_scalar("semff/DELTA3", semff_results["DELTA3"], e)
            writer.add_scalar("semff/lr", lr, e)
        elif isinstance(model.module, net.Hu):
            writer.add_scalar("data/RMSE", results["RMSE"], e)
            writer.add_scalar("data/ABS_REL", results["ABS_REL"], e)
            writer.add_scalar("data/LG10", results["LG10"], e)
            writer.add_scalar("data/DELTA1", results["DELTA1"], e)
            writer.add_scalar("data/DELTA2", results["DELTA2"], e)
            writer.add_scalar("data/DELTA3", results["DELTA3"], e)
            writer.add_scalar("data/lr", lr, e)

        save_checkpoint(model.state_dict(), e, out_dir)


def train(train_loader, model, optimizer, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        iteration_num = (epoch - 1) * len(train_loader) + i

        image, depth = sample_batched['image'], sample_batched['depth']

        image = image.type(torch.float32)
        depth = depth.type(torch.float32)
        depth = depth.cuda(async=True)
        image = image.cuda()

        image = torch.autograd.Variable(image, volatile=True)
        depth = torch.autograd.Variable(depth, volatile=True)

        optimizer.zero_grad()

        loss = None
        if isinstance(model.module, net.TBDPNet):
            mff_output, semff_output = model(image)

            mff_loss, semff_loss = triple_loss([mff_output, semff_output], depth,
                                               cos, get_gradient, iteration_num, writer)

            mutual_depth_loss = torch.log(torch.abs(mff_output - semff_output) + 0.5).mean()
            mutual_loss = mutual_depth_loss
            writer.add_scalar('loss/mutual_loss', mutual_loss.item(), iteration_num)
            writer.add_scalar('loss_terms/mutual_depth', mutual_depth_loss.item(), iteration_num)

            loss = mff_loss + semff_loss + mutual_loss * 0.5

            # ReLU Mutual Loss
            """
            mff_diff = torch.abs(mff_output - depth)
            semff_diff = torch.abs(semff_output - depth)
            
            mff_mutual_diff = mff_diff - semff_diff
            mff_mask = mff_mutual_diff > 0
            mff_mutual_diff = torch.log(F.relu(mff_mutual_diff) + 0.5) * mff_mask.float()
            mff_mutual_loss = (mff_mutual_diff.sum() / mff_mask.sum().float())

            semff_mutual_diff = semff_diff - mff_diff
            semff_mask = semff_mutual_diff > 0
            semff_mutual_diff = torch.log(F.relu(semff_mutual_diff) + 0.5) * semff_mask.float()
            semff_mutual_loss = (semff_mutual_diff.sum() / semff_mask.sum().float())
            
            writer.add_scalar('loss_terms/mff_mutual', mff_mutual_loss.item(), iteration_num)
            writer.add_scalar('loss_terms/wmff_mutual', semff_mutual_loss.item(), iteration_num)
            loss = mff_loss + semff_loss + mff_mutual_loss * 0.5 + semff_mutual_loss * 0.5
            """

            writer.add_scalar('loss/main', loss.item(), iteration_num)
        elif isinstance(model.module, net.Hu):
            output = model(image)

            loss = triple_loss(output, depth,
                               cos, get_gradient, iteration_num, writer)

        losses.update(loss.data[0], image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
   
        batchSize = depth.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})'
          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))


def triple_loss(preds, gt, cos, get_gradient, iteration_num, writer):
    """
    Calculating Losses for each prediction in preds
    :param preds:  [Tensor, Tensor] or Tensor
    :param gt: ground truth tensor
    :param cos: function for calculating normal
    :param get_gradient: class for gradient calculation
    :param iteration_num: iteration number for plotting loss terms to tensorboardX
    :param writer: tensorboardX's writer
    :return: losses [Tensor, Tensor] or Tensor
    """

    ones = torch.ones(gt.size(0), 1, gt.size(2), gt.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)

    gt_grad = get_gradient(gt)
    gt_grad_dx = gt_grad[:, 0, :, :].contiguous().view_as(gt)
    gt_grad_dy = gt_grad[:, 1, :, :].contiguous().view_as(gt)

    gt_normal = torch.cat((-gt_grad_dx, -gt_grad_dy, ones), 1)

    if isinstance(preds, torch.Tensor):
        preds = [preds]

    losses = []
    for i, pred in enumerate(preds):
        pred_grad = get_gradient(pred)
        pred_grad_dx = pred_grad[:, 0, :, :].contiguous().view_as(gt)
        pred_grad_dy = pred_grad[:, 1, :, :].contiguous().view_as(gt)
        pred_normal = torch.cat((-pred_grad_dx, -pred_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(pred - gt) + 0.5).mean()
        loss_dx = torch.log(torch.abs(pred_grad_dx - gt_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(pred_grad_dy - gt_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(pred_normal, gt_normal)).mean()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        if i == 0:
            if len(preds) == 2:
                writer.add_scalar('loss/mff_loss', loss.item(), iteration_num)
                writer.add_scalar('loss_terms/mff_depth', loss_depth.item(), iteration_num)
                writer.add_scalar('loss_terms/mff_grad', (loss_dx + loss_dy).item(), iteration_num)
                writer.add_scalar('loss_terms/mff_normal', loss_normal.item(), iteration_num)
            else:
                writer.add_scalar('loss/loss', loss.item(), iteration_num)
                writer.add_scalar('loss/loss_depth', loss_depth.item(), iteration_num)
                writer.add_scalar('loss/loss_grad', (loss_dx + loss_dy).item(), iteration_num)
                writer.add_scalar('loss/loss_normal', loss_normal.item(), iteration_num)

        elif i == 1:
            writer.add_scalar('loss/wmff_loss', loss.item(), iteration_num)
            writer.add_scalar('loss_terms/wmff_depth', loss_depth.item(), iteration_num)
            writer.add_scalar('loss_terms/wmff_grad', (loss_dx + loss_dy).item(), iteration_num)
            writer.add_scalar('loss_terms/wmff_normal', loss_normal.item(), iteration_num)

        losses.append(loss)

    if len(losses) == 2:
        return losses
    else:
        return losses[0]



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, i, out_dir):
    directory = os.path.join(out_dir, "checkpoints")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = osp.join(directory, "checkpoint_" + str(i) + ".path.tar")
    torch.save(state, filename)


if __name__ == '__main__':
    main()
