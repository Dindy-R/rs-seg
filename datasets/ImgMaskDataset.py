'''
@Description: 用于自定义数据集的pytorch Dataset示例
'''
import os
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMG_FORMAT = ['.jpg', '.png', '.tif']

# class OwnDataset(Dataset):
#     def __init__(self, lines, train, dataset_dir, transform):
#         self.lines = lines
#         self.train = train
#         self.list = len(lines)
#         self.dataset_dir = dataset_dir
#         self.transform = transform
#
#     def __len__(self):
#         return self.list
#
#     def __getitem__(self, index):
#         lines = self.lines[index]
#         name = lines.split()[0]
#
#         image = cv2.imread(os.path.join(os.path.join(self.dataset_dir, "images"), name + ".tif"),
#                            cv2.COLOR_BGR2RGB)
#         label = cv2.imread(os.path.join(os.path.join(self.dataset_dir, "labels"), name + ".png"),
#                            cv2.IMREAD_GRAYSCALE)
#         # print(type(label), isinstance(label, np.ndarray))
#         transformed = self.transform(image=image, label=label)
#
#         image = transformed['image']
#         label = transformed['label']
#         # label = label[None, :, :]
#         label = label.astype(np.int64)
#
#         return {
#             'image': image,
#             'label': label
#         }

class OwnDataset(Dataset):
    def __init__(self, lines, train, dataset_dir, transform):
        self.lines = lines
        self.train = train
        self.list = len(lines)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return self.list

    def __getitem__(self, index):
        lines = self.lines[index]
        name = lines.split()[0]
        image_paths = glob.glob(os.path.join(self.dataset_dir, "images", name + ".*"))
        image_path = image_paths[0]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask_paths = glob.glob(os.path.join(self.dataset_dir, "labels", name + ".*"))
        mask_path = mask_paths[0]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # resize = A.Resize(224, 224)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        mask = mask[None, :, :]
        return {
            'image': image,
            'label': mask.long()
        }


class DualDataset(Dataset):
    def __init__(self, data_dirs, transform):
        super().__init__()
        self.dataset_urls = data_dirs
        self.transform = transform
        self.dataset = self._load_dataset()
        print(f'> dataset size: {len(self.dataset)}')

    def _load_dataset(self):
        _dataset = []
        for url in self.dataset_urls:
            img_dir = os.path.join(url, 'images')
            label_dir = os.path.join(url, 'labels')
            fids = [f for f in os.listdir(img_dir) if f[-4:] in IMG_FORMAT]
            _dataset.extend([(os.path.join(img_dir, fid),
                              os.path.join(label_dir, fid)) for fid in fids])
        return _dataset

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        image = cv2.cvtColor(cv2.imread(data[0]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(data[1], cv2.IMREAD_GRAYSCALE)
        # print(type(mask), isinstance(mask, np.ndarray))
        # mask[mask > 13] = 13
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        mask = mask[None, :, :]
        return {
            'image': image,
            'label': mask.long()
        }


def get_train_transform():
    return A.Compose([
                A.Resize(224, 224),
                A.Flip(p=0.75),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(p=0.3),
                A.RandomGamma(p=0.3),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1, p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=0),
                A.Normalize(mean=MEAN, std=STD),

                ToTensorV2(),
            ])


def get_val_transform():
    return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=MEAN, std=STD),

                ToTensorV2(),
            ])


if __name__ == '__main__':
    import os
    import cv2
    from torch.utils.data import DataLoader
    '''test DataGenerator'''

    dirs = ["../../DATASET/v1/ccf/train"]
    class_info = be_Class2Id
    transform = get_train_transform()
    BS = 4
    D = batchtest_Dataset(dirs, class_info, transform, target_class='water')
    dataset = D.get_dataset()

    for i in range(10):
        print(dataset[i])

    print(20*'=')

    test_data = batchtest_Dataset(img, tile_size, overlap, transform)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size,
        drop_last=False)

    data_loader = DataLoader(dataset=D, batch_sampler=S, num_workers=1)

    mean = np.array([0.11894194, 0.12947349, 0.1050701])
    std = np.array([0.08124223, 0.09198588, 0.08354711])

    for idx, batch_samples in enumerate(data_loader):
        imgs, labels = batch_samples['image'], batch_samples['label']
        imgs = imgs.numpy()
        labels = labels.numpy()
        print(imgs.shape, labels.shape)

        imgs = np.transpose(imgs, (0, 2, 3, 1))
        labels = np.transpose(labels, (0, 2, 3, 1))
        for i in range(BS):
            img = imgs[i, :, :, 0:3]
            img = (img * std + mean) * 255
            img = img.astype(np.uint8)
            label = labels[i].astype(np.uint8) * 255
            cv2.imwrite(os.path.join('test', 'sample_b{:03d}_{:02d}.tif'.format(idx, i)), img)
            cv2.imwrite(os.path.join('test', 'sample_b{:03d}_{:02d}.png'.format(idx, i)), label)

        if idx >= 10: break
        