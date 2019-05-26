import argparse
import pickle
from utils import util
from utils.visualize_util import *
from tqdm import tqdm

from dataset import loaddata
from dataset import loaddata_sun
import numpy as np
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
from utils import sobel

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--model', default="tbdp", type=str,
                    help='used model type ["tbdp", "hu"]')
parser.add_argument('--semff', action="store_true",
                    help='Use SE_MFF Module')
parser.add_argument('--pcamff', action="store_true",
                    help='Use PCAMFF Module')
parser.add_argument('--parallel', action="store_true",
                    help='Use parallel R')
parser.add_argument('--dataset', default="nyud", type=str,
                    help='used dataset ["nyud", "sun"]')

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

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True,
                         model=args.model, parallel=args.parallel, semff=args.semff, pcamff=args.pcamff)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(
        torch.load('/home/takagi.kazunari/projects/TBDP-Net/runs/1231_TSM_PCAMFFv2_b16_step3/checkpoints/checkpoint_16.path.tar'))

    if args.dataset == "nyud":
        test_loader = loaddata.getTestingData(nyud_root_path, 16)
    elif args.dataset == "sun":
        test_loader = loaddata_sun.getTestingData(sun_root_path, 16)
    else:
        raise NotImplementedError('Specify dataset in [\'nyud\', \'sun\']')

    if args.model == 'tbdp':
        mff_metrics, semff_metrics = test_tbdp(test_loader, model, dataset=args.dataset, returnValue=True)

        print("mff_metrics:")
        print(mff_metrics)
        print("")
        print("semff_metrics:")
        print(semff_metrics)

        with open("/home/takagi.kazunari/projects/TBDP-Net/tmp/tsm_metrics.pkl", "wb") as f:
            out = {}
            out["mff"] = mff_metrics
            out["semff"] = semff_metrics

            pickle.dump(out, f)
    elif args.model == 'hu':
        metrics = test_hu(test_loader, model, dataset=args.dataset, returnValue=True)

        print("metrics:")
        print(metrics)

        with open("/home/takagi.kazunari/projects/TBDP-Net/tmp/tsm_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)


def test_tbdp(test_loader, model, dataset='nyud', returnValue=False, returnSamples=False, sample_idx=[0]):
    model.eval()

    gt_depths = None
    mff_depths = None
    semff_depths = None

    image_samples = []
    mff_depth_samples = []
    semff_depth_samples = []

    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(test_loader)):
            image, depth = sample_batched['image'], sample_batched['depth']

            image = image.type(torch.float32)
            depth = depth.type(torch.float32)
            depth = depth.cuda(async=True)
            image = image.cuda()

            image = torch.autograd.Variable(image, volatile=True)
            depth = torch.autograd.Variable(depth, volatile=True)

            mff_output, semff_output = model(image)
            mff_output = torch.nn.functional.upsample(mff_output, size=[depth.size(2), depth.size(3)], mode='bilinear')
            semff_output = torch.nn.functional.upsample(semff_output, size=[depth.size(2), depth.size(3)],
                                                        mode='bilinear')

            if gt_depths is None:
                gt_depths = depth
                mff_depths = mff_output
                semff_depths = semff_output
            else:
                gt_depths = torch.cat((gt_depths, depth), 0)
                mff_depths = torch.cat((mff_depths, mff_output), 0)
                semff_depths = torch.cat((semff_depths, semff_output), 0)

            if returnSamples and i in sample_idx:
                image_samples.append(image[0])
                mff_depth_samples.append(mff_output[0])
                semff_depth_samples.append(semff_output[0])

            torch.cuda.empty_cache()

        if dataset == 'nyud':
            mff_errors = util.evaluateError(mff_depths, gt_depths)
            semff_errors = util.evaluateError(semff_depths, gt_depths)
        elif dataset == 'sun':
            mff_errors = util.evaluateError_M(mff_depths, gt_depths)
            semff_errors = util.evaluateError_M(semff_depths, gt_depths)

        mff_errors['RMSE'] = np.sqrt(mff_errors['MSE'])
        semff_errors['RMSE'] = np.sqrt(semff_errors['MSE'])

        torch.cuda.empty_cache()

        if returnValue and returnSamples:
            return mff_errors, semff_errors, image_samples, mff_depth_samples, semff_depth_samples
        elif returnValue:
            return mff_errors, semff_errors
        elif returnSamples:
            return image_samples, mff_depth_samples, semff_depth_samples


def test_hu(test_loader, model, dataset='nyud', returnValue=False, returnSamples=False, sample_idx=[0]):
    model.eval()

    gt_depths = None
    pred_depths = None

    image_samples = []
    depth_samples = []

    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(test_loader)):
            image, depth = sample_batched['image'], sample_batched['depth']

            image = image.type(torch.float32)
            depth = depth.type(torch.float32)
            depth = depth.cuda(async=True)
            image = image.cuda()

            image = torch.autograd.Variable(image, volatile=True)
            depth = torch.autograd.Variable(depth, volatile=True)

            output = model(image)
            output = torch.nn.functional.upsample(output, size=[depth.size(2), depth.size(3)], mode='bilinear')

            if gt_depths is None:
                gt_depths = depth
                pred_depths = output
            else:
                gt_depths = torch.cat((gt_depths, depth), 0)
                pred_depths = torch.cat((pred_depths, output), 0)

            if returnSamples and i in sample_idx:
                image_samples.append(image[0])
                depth_samples.append(output[0])

            torch.cuda.empty_cache()

        if dataset == 'nyud':
            errors = util.evaluateError(pred_depths, gt_depths)
        elif dataset == 'sun':
            errors = util.evaluateError_M(pred_depths, gt_depths)

        errors['RMSE'] = np.sqrt(errors['MSE'])

        torch.cuda.empty_cache()

        if returnValue and returnSamples:
            return errors, image_samples, depth_samples
        elif returnValue:
            return errors
        elif returnSamples:
            return image_samples, depth_samples


def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
                 torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
