import os
import sys
from datetime import datetime

from configs.paths import model_dir

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as T
from torchsummary import summary
import torch.nn.functional as F
# MAKE SURE TO SHUFFLE IMPORT ORDER AND DELETE USELESS IMPORTS
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout, BatchNorm2d, BCELoss
from torch.optim import Adam, SGD


def test(model, dataloaders, criterion):


    testloader = dataloaders["test"]
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        for data in testloader:
            images, labels = data
            images = images.float()
            labels = labels.long()

            preds = model(images)
            preds = torch.argmax(preds, 1)

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)




        acc = accuracy_score(all_labels, all_preds)
        print("Accuracy: {}".format(acc))
        return (all_labels, all_preds)