import os
import sys
from datetime import datetime

from configs.paths import model_dir

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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



def train(model, dataloaders, epochs, optimizer, criterion, validation=True, save_trained=True, save_name=""):

    train_losses = []
    
    
    train_loader = dataloaders["train"]
    if validation:
        validation_losses = []
        lowest_val_loss = np.inf
        val_loader = dataloaders["validation"]

    
    for epoch in range(epochs):

        train_loss = 0
        n_batches = 0

        model.train()
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()

            optimizer.zero_grad()
            output_train = model(inputs)
            loss = criterion(output_train, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            n_batches += 1
        
        train_loss = train_loss/n_batches
        train_losses.append(train_loss)

        
        if validation:
            model.eval()

            best_model = None
            validation_loss = 0
            n_batches = 0
            num_correct = 0
            total_data_len = 0

            for i, data in enumerate(val_loader):

                inputs, labels = data
                inputs = inputs.float()
                labels = labels.long()

                val_output = model(inputs)
                loss = criterion(val_output, labels)

                validation_loss += loss.item()
                n_batches += 1

                num_correct += (torch.argmax(val_output, 1) == labels).float().sum()
                total_data_len += len(labels)
            
            validation_loss = validation_loss/n_batches
            if validation_loss < lowest_val_loss:
                best_model = model.state_dict()
                lowest_val_loss = validation_loss
                
            validation_losses.append(validation_loss)
            validation_accuracy = num_correct/total_data_len

        print("Epoch: {}  Training Loss: {}  Validation Loss: {}  Validation Acc: {}  Validation Data len: {}"
                .format(epoch, train_loss, validation_loss, validation_accuracy, total_data_len))

    if save_trained:
        
        if save_name:
            fname = save_name + ".pth"
        else:
            datetimeobj = datetime.now()
            tmo = datetimeobj.time()
            fname = "{}_{}_{}.pth".format(tmo.hour, tmo.minute, tmo.second)

        if validation:
            to_save = best_model
        else:
            to_save = model.state_dict()

        save_path = os.path.join(model_dir, "saved_models", fname)
        torch.save(to_save, save_path)
        print("Model saved to {}".format(save_path))