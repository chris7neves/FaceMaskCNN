import os
from datetime import datetime

import matplotlib

from configs.paths import model_dir

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from PIL import ImageFile

from datasets import RANDOM_SEED
ImageFile.LOAD_TRUNCATED_IMAGES=True
import torch

RANDOM_SEED = 42

def train_val(model, dataloaders, epochs, optimizer, criterion, validation=True, save_name=""):
    """
    The main training loop for the model.
    :model: the model to be trained
    :dataloaders: the dataloader dict, with the training loader ready
    :epochs: the number of epochs to train the model for
    :optimizer: the optimizer to use during training
    :criterion: the loss criterion
    :validation: if true, performs a validation loop at the end of training and enables early saving
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    model.to(device)

    train_losses = []
    train_accuracies = []
    
    if save_name:
        fname = save_name + ".pth"
    else:
        datetimeobj = datetime.now()
        tmo = datetimeobj.time()
        fname = "{}_{}_{}.pth".format(tmo.hour, tmo.minute, tmo.second)
    save_path = os.path.join(model_dir, "saved_models", fname)
    
    train_loader = dataloaders["train"]
    if validation:
        validation_losses = []
        validation_accuracies = []
        lowest_val_loss = np.inf
        val_loader = dataloaders["validation"]
    
    # Training loop
    for epoch in range(epochs):

        train_loss = 0
        n_batches = 0
        num_correct = 0
        total_data_len = 0

        model.train()
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.float()
            labels = labels.long()


            #show_batch(inputs, labels)

            optimizer.zero_grad()
            output_train = model(inputs)
            loss = criterion(output_train, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_correct += (torch.argmax(output_train, 1) == labels).float().sum()
            total_data_len += len(labels)

            n_batches += 1
        
        train_loss = train_loss/n_batches
        train_losses.append(train_loss)
        train_accuracy = num_correct/total_data_len
        train_accuracies.append(train_accuracy)
        
        # Validation loop
        if validation:
            model.eval()

            best_model = None
            validation_loss = 0
            n_batches = 0
            num_correct = 0
            total_data_len = 0

            for i, data in enumerate(val_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                inputs = inputs.float()
                labels = labels.long()

                val_output = model(inputs)
                loss = criterion(val_output, labels)

                validation_loss += loss.item()
                n_batches += 1

                num_correct += (torch.argmax(val_output, 1) == labels).float().sum()
                total_data_len += len(labels)
            
            validation_loss = validation_loss/n_batches

            # If current model performs better than previous validation iterations, save it
            if validation_loss < lowest_val_loss:
                print("New lowest validation loss. Saving model...")
                lowest_val_loss = validation_loss
                to_save = model.state_dict()
                torch.save(to_save, save_path)
                print("Model saved to {}".format(save_path))

            # Record validation losses and accuracy
            validation_losses.append(validation_loss)
            validation_accuracy = num_correct/total_data_len
            validation_accuracies.append(validation_accuracy)

        dtobj = datetime.now().time()
        if validation:
            print(dtobj, " ||  Epoch: {}  Training Loss: {}  Validation Loss: {}  Validation Acc: {}  Validation Data len: {}"
                    .format(epoch, train_loss, validation_loss, validation_accuracy, total_data_len))
        else:
            print(dtobj, " ||  Epoch: {}  Training Loss: {}  Training Accuracy: {}"
                    .format(epoch, train_loss, train_accuracy))

    # If validation not enabled, simply save the model that was just trained.
    if not validation:
        to_save = model.state_dict()
        torch.save(to_save, save_path)
        print("Model saved to {}".format(save_path))

    if validation:
        return (train_losses, validation_losses, validation_accuracies)
    else:
        return (train_losses, None, None)


        