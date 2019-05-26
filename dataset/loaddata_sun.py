import glob
import os.path as osp

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.sun_transform import *


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, mode="train"):
        paths = sorted(glob.glob(osp.join(root_dir, mode, "npy", "*")))
        self.paths = paths

        self.transform = transform

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])

        tmp_img = data[:3, :, :].copy()
        tmp_img[0, :, :] = data[2, :, :]
        tmp_img[2, :, :] = data[0, :, :]
        image = Image.fromarray(np.asarray(tmp_img.transpose(1, 2, 0), dtype=np.uint8), "RGB")

        depth = Image.fromarray(data[3, :, :], "F")

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.paths)


def getTrainingData(dir_path, batch_size=64):
    __sunrgbd_stats = {'mean': [0.494, 0.457, 0.433],
                       'std': [0.256, 0.261, 0.264]}

    transformed_training = depthDataset(root_dir=dir_path,
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__sunrgbd_stats['mean'],
                                                      __sunrgbd_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=1, pin_memory=False)

    return dataloader_training


def getTestingData(dir_path, batch_size=64):

    #__sunrgbd_stats = {'mean': [0.494, 0.457, 0.433],
    #                   'std': [0.256, 0.261, 0.264]}
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(root_dir=dir_path,
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]), mode="test")

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=1, pin_memory=False)

    return dataloader_testing
