from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_confusion_matrix_df(labels, preds, label_dict):
    """Generates a confusion matrix in dataframe form given the true labels and predictions."""  
    cm_df = pd.DataFrame(confusion_matrix(labels, preds), 
                    index=[i[1] for i in label_dict.items()], 
                    columns=[i[1] for i in label_dict.items()])
    return cm_df

def get_confusion_matrix_ax(cm_df, figsize=(15, 8)):
    """Gets the ax object of the confusion matrix. Used for plotting."""
    # Plot inspiration from  https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    sns.set(rc = {'figure.figsize':figsize})
    ax = sns.heatmap(cm_df, annot=True, fmt="d")
    return ax

def get_train_val_curve(train_loss, val_loss=None, val_acc=None, figsize=(15, 8)):
    """Given the training and validation losses, generate the training, validation loss and validation accuracy curve."""
    fig = plt.figure(figsize=figsize)
    x_axis = np.arange((len(train_loss)))
    plt.plot(x_axis, train_loss, label="Training Loss")
    if val_loss:
        plt.plot(x_axis, val_loss, label="Validation Loss")
        plt.plot(x_axis, val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses per Epoch")

    return fig

def get_f1_score(labels, preds, average='micro'):
    """Get the F1 score for the predictions."""
    f1 = f1_score(labels, preds, average=average)
    return f1

def get_accuracy(labels, preds, raw_correct=True):
    """Get the accuracy of the predictions. If raw correct, simply return number of correct predictions."""
    return accuracy_score(labels, preds, normalize=raw_correct)

def get_precision_recall(labels, preds, average="micro"):
    """Get the precision and recall of the model predictions."""
    recall = recall_score(labels, preds, average=average)
    precision = precision_score(labels, preds, average=average)
    return (precision, recall)
    

def get_classification_report_df(labels, preds, class_names:list):
    """Get the sklearn classification report. This does some heavy lifting and makes metric reporting easier."""
    cr = classification_report(labels, preds, target_names=class_names, output_dict=True)
    cr_final = {}
    for k, v in cr.items():
        inner = v
        if(isinstance(v, dict)):
            inner = {}
            for k2, v2 in v.items():
                if k2 == "support":
                    inner["Num_Samples"] = int(v2)
                else:
                    inner[k2] = round(v2, 4)
        cr_final[k] = inner
    cr_df = pd.DataFrame.from_dict(cr_final, orient="columns")
    return cr_df

