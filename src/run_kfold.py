from audioop import bias
from models.available_models import model_dict
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
import torch
import metrics_and_plotting as mp
from datetime import datetime


RANDOM_SEED = 42

def run_kfold(model_name, dataset, epochs, batch_sz, n_splits=10):
    """
    Structure of training loop when using kfold a torch.dataset taken from:
    https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
    """

    # Get stratified kfold object
    folded = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for Kfold CV: {}".format(device))

    fold_info = {}
    fold_labels_biases = {}

    for fold, (train_id, val_id) in enumerate(folded.split(np.arange(len(dataset)))):

        print("----- Fold #: {}/{} Train len: {}  Val len: {} -----".format(fold+1, n_splits, len(train_id), len(val_id)))

        train_sampler = SubsetRandomSampler(train_id)
        val_sampler = SubsetRandomSampler(val_id)

        train_loader = DataLoader(dataset, batch_size=batch_sz, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_sz, sampler=val_sampler)
        
        model_details = model_dict[model_name]()
        model = model_details["model"]
        model.to(device)
        optimizer = model_details["optimizer"]
        criterion = model_details["criterion"]

        all_preds = []
        all_labels = []
        all_biases = []
        info = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

        # Train loop
        model.train()
        for epoch in range(epochs):

            n_batches = 0
            train_loss, num_correct, total_data_len = 0.0, 0, 0
            for images, package in train_loader:

                if len(package) == 2:
                    labels = package[0]
                    #biases = package[1]
                else:
                    labels = package

                images, labels = images.to(device), labels.to(device)
                images = images.float()
                labels = labels.long()

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_correct += (torch.argmax(output, 1) == labels).float().sum()
                total_data_len += len(labels) 

                n_batches += 1

            train_loss = train_loss/n_batches
            info["train_loss"].append(train_loss)
            train_accuracy = num_correct.item()/total_data_len
            info["train_acc"].append(train_accuracy)


            # Valid loop
            model.eval()
            n_batches = 0
            valid_loss, num_correct, total_data_len = 0.0, 0, 0

            for images, package in val_loader:
                
                if len(package) == 2:
                    labels = package[0]
                    biases = package[1]
                else:
                    labels = package

                images, labels = images.to(device), labels.to(device)
                images = images.float()
                labels = labels.long()

                output = model(images)
                loss = criterion(output, labels)

                valid_loss += loss.item()
                n_batches += 1

                num_correct += (torch.argmax(output, 1) == labels).float().sum()
                total_data_len += len(labels)

                if output.is_cuda and labels.is_cuda:
                    outputcpu = output.cpu()
                    labelscpu = labels.cpu()

                if epoch == (epochs-1): # On the final training epoch, record validation results
                    preds = torch.argmax(outputcpu, 1)
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labelscpu.tolist())
                    all_biases.extend(biases)

            valid_loss = valid_loss/n_batches
            info["valid_loss"].append(valid_loss)
            valid_accuracy = num_correct.item()/total_data_len
            info["valid_acc"].append(valid_accuracy)

            dtobj = datetime.now().time()
            print("[{}]:  ||  Epoch: {}  |  Training Loss: {} - Validation Loss: {} - Validation Acc: {} "
                    .format(dtobj, epoch, train_loss, valid_loss, valid_accuracy, total_data_len))
        
        fold_info[str(fold+1)] = info
        results_dict = {"preds": all_preds, "labels": all_labels, "biases": all_biases}
        fold_labels_biases[str(fold+1)] = results_dict

    
    return fold_info, fold_labels_biases
