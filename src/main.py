import argparse
from datetime import datetime
import os
import sys

import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from models.available_models import model_dict
from datasets import get_dataloaders, get_masktype_data_df, get_masktype_datasets, lazy_load_train_val_test, prepare_kfold_strategy
from datasets import prepare_train_val_strategy
import metrics_and_plotting as mp
from train import train_val
from test import test
from run_kfold import run_kfold
import infer as infer
from configs.paths import paths_aug, model_dir, report_dir, get_paths
from generate_report import generate_html_report, generate_html_bias_report
from util import class_dict_from_aug_paths  

####################################################
#                    PARSE ARGS                    #
####################################################

parser = argparse.ArgumentParser(description="Entrypoint to FaceMaskCNN model.")

subparsers = parser.add_subparsers(dest='mode')

# List model parser
list_parser = subparsers.add_parser("model_list")

# Training sub parser
train_parser = subparsers.add_parser("train")
train_parser.add_argument("model_name", action="store")
train_parser.add_argument("--save_losses", action="store_true", default=True)
train_parser.add_argument("--num_epochs", action="store", default=25)
train_parser.add_argument("--train_batchsz", action="store", default=128)
train_parser.add_argument("--val_batchsz", action="store", default=128)
train_parser.add_argument("--skip_val", action="store_true")
train_parser.add_argument("--train_on_everything", action="store_true")

# KFold evaluation sub parser
kfold_parser = subparsers.add_parser("kfold")
kfold_parser.add_argument("model_name", action="store")
kfold_parser.add_argument("paths_json_name", action="store")
kfold_parser.add_argument("--search_subdir", action="store_true")
kfold_parser.add_argument("--bias", action="store_true")
kfold_parser.add_argument("--folds", action="store", default=10)
kfold_parser.add_argument("--num_epochs", action="store", default=25)
kfold_parser.add_argument("--batchsz", action="store", default=128)


# Testing sub parser
test_parser = subparsers.add_parser("test")
test_parser.add_argument("model_name", action="store")
test_parser.add_argument("from_saved", action="store")
test_parser.add_argument("--test_batchsz", action="store", default=300)
test_parser.add_argument("--gen_report", action="store_true")

# Inference sub parser
infer_parser = subparsers.add_parser("infer")
infer_parser.add_argument("img_path", action="store")
infer_parser.add_argument("model_name", action="store")
infer_parser.add_argument("from_saved", action="store")

args = parser.parse_args()
print(args)
####################################################
#                   ARG HANDLING                   #
####################################################

if args.mode == "model_list":
    
    print("Models available to use:")
    for k in model_dict.keys():
        print(k) 

elif args.mode == "train":
    
    model_name = args.model_name

    # Get the model and all its parameters according to the name given in args
    model_details = model_dict[model_name]()
    model = model_details["model"]
    optimizer = model_details["optimizer"]
    criterion = model_details["criterion"]
    transforms = model_details["transforms"]["train"]

    train_on_everything = args.train_on_everything
    skip_val = args.skip_val
    if (not skip_val) and args.train_on_everything:
        print("train_on_everything specified. Skipping validation during training.")
        skip_val=True

    # Prepare dataloaders for train/validation set strategy
    dataloaders = prepare_train_val_strategy(get_paths("paths_aug_balanced_race"), transforms, 
                                            train_on_everything=args.train_on_everything,
                                            skip_val=args.skip_val, search_subdir=True)

    print("\n===============================================================================================")
    print("Begin training - Model: {} - Num Epochs: {}".format(model_name, args.num_epochs))
    print("===============================================================================================\n")
    
    train_losses, validation_losses, validation_accuracies = train_val(model, dataloaders, 
                                                                        int(args.num_epochs), optimizer, 
                                                                        criterion, validation=not skip_val)
    mp.get_train_val_curve(train_losses, validation_losses, validation_accuracies)
    plt.tight_layout()
    plt.show()

    # If save losses is specified, save the train, val loss accuracy curve to the report directory
    if args.save_losses:
        time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        tdf = pd.DataFrame(train_losses)
        tdf.to_csv(
            os.path.join(
                report_dir, "{}_{}_training_losses.csv".format(time_string, model_name))
            )

        vdf = pd.DataFrame(validation_losses)
        vdf.to_csv(
            os.path.join(
                report_dir, "{}_{}_validation_losses.csv".format(time_string, model_name))
            )
        plt.savefig(os.path.join(report_dir, "{}_{}_loss_acc_curve.jpg".format(time_string, model_name)))

elif args.mode == "kfold":

    model_name = args.model_name
    num_epochs = int(args.num_epochs)
    folds = int(args.folds)
    batch_sz = int(args.batchsz)

    # Get the model and all its parameters according to the name given in args
    model_details = model_dict[model_name]()
    transforms = model_details["transforms"]["train"]

    # Get the folder paths that contain the images
    paths = get_paths(args.paths_json_name)

    datasets, label_dict = prepare_kfold_strategy(paths, transforms, args.bias, search_subdir=args.search_subdir, grayscale=False)

    # Results dict contains preds, labels, biases in that order
    # Fold info contains train and val loss and accuracy for each fold
    fold_info, fold_labels_biases  = run_kfold(model_name, datasets["train"], num_epochs, batch_sz, folds)

    generate_html_bias_report(fold_labels_biases, label_dict, model_name, fold_info)

elif args.mode == "test":
    
    # Get the savename of the saved weights and parameters
    savename = args.from_saved

    # Validate savename to make sure it exists before executing code
    saved_model_path = os.path.join(model_dir, "saved_models", savename)
    if os.path.isfile(saved_model_path):
        model_details = model_dict[args.model_name]()
        model = model_details["model"]
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print("Invalid .pth or .pt file specified: {}".format(saved_model_path))
        sys.exit(0)

    data_df = get_masktype_data_df(paths_aug)

    label_dict = dict(list(data_df.groupby(["label_literal", "label"]).indices.keys()))
    label_dict = {v : k for k, v in label_dict.items()}
    labels = data_df.pop("label")

    data_dict = lazy_load_train_val_test(data_df, labels, 0.7, 0.2)
    datasets = get_masktype_datasets(data_dict, model_details["transforms"]["test"], grayscale=False)
    dataloaders = get_dataloaders(datasets, test_batch_size=300)
    criterion = model_details["criterion"]

    test_labels, test_preds = test(model, dataloaders, criterion)

    f1 = mp.get_f1_score(test_labels, test_preds, average=None)
    print("===========================================================")
    if not isinstance(f1, float):
        for i, f in enumerate(f1):
            print("{} F1 Score: {}".format(label_dict[i], f))
    else:
        print("F1 Score: {}".format(f1))
    acc = accuracy_score(test_labels, test_preds)
    print("Accuracy: {}".format(acc))
    num_corr = mp.get_accuracy(test_labels, test_preds, True)
    print("{} correct out of {}".format(num_corr, len(test_labels)))
    print("===========================================================")

    if args.gen_report:
        generate_html_report(test_labels, test_preds, label_dict, 
                            model_name=model.__class__.__name__,
                            model_param_file=savename,  
                            dest=report_dir)

elif args.mode == "infer":

    img_path = args.img_path
    model_name = args.model_name
    param_file = args.from_saved

    # Open the model with the parameters passed through the command line
    saved_model_path = os.path.join(model_dir, "saved_models", param_file)
    if os.path.isfile(saved_model_path):
        model_details = model_dict[model_name]()
        model = model_details["model"]
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print("Invalid .pth or .pt file specified: {}".format(saved_model_path))
        sys.exit(0)

    # Run the inference
    transform = model_details["transforms"]["test"]
    probs, preds = infer.infer(img_path, model, transform, as_label=True, label_dict=class_dict_from_aug_paths())

    # Display the image along with the results of the inference
    to_tensor = T.Compose([T.ToTensor()])
    image = infer.prep_image(img_path, transforms=to_tensor)

    print("Class: {}".format(preds))
    print("Probabilities: {}".format(probs.tolist()))

    plt.imshow(image[0].permute(1, 2, 0))
    plt.title(preds)
    plt.show()