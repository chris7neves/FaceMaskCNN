import os
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

import matplotlib.pyplot as plt


class LeafData(Dataset):

    def __init__(self,
                 data,
                 directory,
                 transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # import
        path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


if __name__=="__main__":
    device      = torch.device('cpu')
    num_workers = 4
    image_size  = 512
    batch_size  = 8
    data_path   = '../cropped/'

    df = pd.read_csv(data_path + 'image_ids.csv')
    df.head()

    augs = A.Compose([A.Resize(height=image_size,
                               width=image_size),
                      A.Normalize(mean=(0, 0, 0),
                                  std=(1, 1, 1)),
                      ToTensorV2()])


    # dataset
    image_dataset = LeafData(data=df,
                             directory=data_path + 'all_images/',
                             transform=augs)

    # data loader
    image_loader = DataLoader(image_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)

    # display images
    # for batch_idx, inputs in enumerate(image_loader):
    #     fig = plt.figure(figsize=(14, 7))
    #     for i in range(8):
    #         ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    #         plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
    #     break
    #
    # plt.show()

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    count = len(df) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))