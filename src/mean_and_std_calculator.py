import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import albumentations
from albumentations import pytorch as atorch


class ImageDataset(Dataset):
    """ As per https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/ """

    def __init__(self, df, path_to_folder, transforms=None):
        self.df = df
        self.path_to_folder = path_to_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.path_to_folder, self.df.iloc[index]['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)
        return image['image']


def compute_mean_and_std_torchvision(train_loader):
    std = 0
    count = 0
    mean = 0

    for images, _ in train_loader:
        batch_count = images.size(0)
        images = images.view(batch_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        count += batch_count
    mean /= count
    std /= count

    return mean, std


def calculator_torchvision():
    # The full dataset is not pushed to the repo due to its size
    data_path = '../images/'
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=12,
                              shuffle=False)

    mean, std = compute_mean_and_std_torchvision(train_loader)
    print('mean: ' + str(mean))
    print('std:  ' + str(std))


def _get_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(64, 64),
            albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            atorch.ToTensorV2(),
        ]
    )

def _get_per_channel_sums(dataloader):
    default_tensor = [0.0, 0.0, 0.0]
    axis = [0, 2, 3]

    pixel_sum_per_channel = torch.tensor(default_tensor)
    pixel_sum_per_channel_squared = torch.tensor(default_tensor)

    for tensor in dataloader:
        pixel_sum_per_channel += tensor.sum(axis=axis)
        pixel_sum_per_channel_squared += (tensor.data ** 2).sum(axis=axis)

    return pixel_sum_per_channel, pixel_sum_per_channel_squared

def _get_mean_and_std(df, pixel_sum_per_channel, pixel_sum_per_channel_squared):
    """ As per https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266 """
    total_pixels = len(df) * (64**2)

    mean = pixel_sum_per_channel / total_pixels
    variance = (pixel_sum_per_channel_squared / total_pixels) - (mean**2)
    std = torch.sqrt(variance)

    return mean, std

def calculator_albumentations():
    # Not loaded to the repo due to size considerations
    image_directory = '../cropped'
    # Pregenerated CSV with all filenames
    df = pd.read_csv(image_directory + '/filenames.csv')

    images = ImageDataset(df=df, path_to_folder=image_directory + '/all_images/', transforms=_get_transforms())
    dataloader = DataLoader(dataset=images, batch_size=32, shuffle=False)

    pixel_sum_per_channel, pixel_sum_per_channel_squared = _get_per_channel_sums(dataloader)
    mean, std = _get_mean_and_std(df, pixel_sum_per_channel, pixel_sum_per_channel_squared)

    print('Mean: ' + str(mean))
    print('Standard deviation:  ' + str(std))


if __name__=="__main__":
    calculator_albumentations()