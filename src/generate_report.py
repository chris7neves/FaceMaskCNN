import metrics_and_plotting as mp
from configs.paths import report_dir
import matplotlib.pyplot as plt

import pandas as pd
import os
from datetime import datetime

def generate_html_report(labels, preds, class_dict, model_name, model_param_file, dest=report_dir):
    """
    Prepare a html report for the output of the model testing.
    labels: true labels, categorical (not binary)
    preds: predictions of the model, after going through argmax
    class_dict: dict linking cat labels to string labels
    model_name: name of the model that generated the predictions
    model_param_file: .pth of the parameters use to get predictions
    """

    accuracy = mp.get_accuracy(labels, preds)
    micro_precision, micro_recall = mp.get_precision_recall(labels, preds, average="micro")
    micro_f1 = mp.get_f1_score(labels, preds, average="micro")

    cf_matrix_path = os.path.join(dest, "cf_matrix.jpg")
    ax = mp.get_confusion_matrix_ax(mp.get_confusion_matrix_df(labels, preds, class_dict))
    ax.figure.savefig(cf_matrix_path)

    c_report = mp.get_classification_report_df(labels, preds, class_names=class_dict.values())
    
    html_testing = '''

    <html>
        <head>
            <title>{page_title}</title>
        </head>
        <body>
            <h1>{model_string}</h1>
            <h3>Report generated on: {date_time}</h3>
            <h2> Global/Micro overview </h2>
            <p> F1 Micro: {f1_micro}</p>
            <p> Precision Micro: {precision_micro}</p>
            <p> Recall Micro: {recall_micro}</p>
            <p> Global Accuracy: {accuracy}</p>
            <h2> Per class metrics </h2>
            {classification_report}
            <h2> Confusion Matrix </h2>
            <img src='{confusion_matrix_path}' width="850">

        </body>
    </html>
    '''.format(
        page_title="Test Performance Report: {}".format(model_name),
        model_string="Model: {} | Parameter File: {}".format(model_name, model_param_file),
        date_time=datetime.now(),
        f1_micro=micro_f1,
        precision_micro=micro_precision,
        recall_micro=micro_recall,
        accuracy=accuracy,
        classification_report=c_report.to_html(),
        confusion_matrix_path=cf_matrix_path
    )

    save_path = os.path.join(dest, '{}_test_report.html'.format(model_param_file.split('.')[0]))
    with open(save_path, 'w') as f:
        f.write(html_testing)

def generate_html_bias_report(fold_labels_biases, label_dict, model_name, fold_info, dest=report_dir):
    """
    Prepare a html report for the output of the model testing.
    labels: true labels, categorical (not binary)
    preds: predictions of the model, after going through argmax
    class_dict: dict linking cat labels to string labels
    model_name: name of the model that generated the predictions
    """

    # Get the global results over all folds
    global_preds = []
    global_labels = []
    global_biases = []
    for fold, results_dict in fold_labels_biases.items():
        global_labels.extend(results_dict["labels"])
        global_preds.extend(results_dict["preds"])
        global_biases.extend(results_dict["biases"])

    # Label distribution per bias category
    df_distr = mp.get_class_distributions_bias(global_labels, global_biases, label_dict)
    
    # Per fold accuracy, f1 score, precision, recall as well as per fold metrics per bias
    df_pfmet = pd.DataFrame(columns=["Fold", "Acc", "F1_macro", "Prec_macro", "Recall_macro"])
    for f, results in fold_labels_biases.items():
        acc = mp.get_accuracy(results["labels"], results["preds"])
        f1 = mp.get_f1_score(results["labels"], results["preds"], average="weighted")
        prec, rec = mp.get_precision_recall(results["labels"], results["preds"], average="weighted")
        temp = {"Fold":f, 
                "Acc":acc, 
                "F1_macro":f1, 
                "Prec_macro":prec, 
                "Recall_macro":rec}
        df_pfmet = df_pfmet.append(temp, ignore_index=True)

    # Global accuracy, f1 score, precision, recall (mean of metrics per fold, and their std) 
    df_stats = pd.DataFrame(columns=df_pfmet.drop("Fold", axis=1).columns)
    df_stats.loc["Mean"] = df_pfmet.drop("Fold", axis=1).mean()
    df_stats.loc["Std"] = df_pfmet.drop("Fold", axis=1).std()

    # Per class metrics for all folds
    # print("Label length in metrics: {}".format(len(global_labels)))
    # print("Preds length in metrics: {}".format(len(global_preds)))
    c_report = mp.get_classification_report_df(global_labels, global_preds, class_names=label_dict.values())

    # Confusion Matrix over all folds
    time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    cf_matrix_path_all = os.path.join(dest, "{}_cf_matrix_all.jpg".format(time_string))
    ax = mp.get_confusion_matrix_ax(mp.get_confusion_matrix_df(global_labels, global_preds, label_dict))
    ax.figure.savefig(cf_matrix_path_all)
    plt.clf()

    # Global scores for all biases and confusion matrix per bias
    bias_cf_matrix_string = ""
    template = "<p>Subset: {} </p><img src='{}' width='850'>"
    df_bias = pd.DataFrame({"labels":global_labels, "preds":global_preds, "biases":global_biases})
    df_biasstat = pd.DataFrame(columns=["Subset", "Acc", "F1_macro", "Prec_macro", "Recall_macro"])
    for bias in df_bias["biases"].unique():
        labels = df_bias.loc[df_bias["biases"].str.match(bias), "labels"]
        preds = df_bias.loc[df_bias["biases"].str.match(bias), "preds"]
        
        acc = mp.get_accuracy(labels, preds)
        f1 = mp.get_f1_score(labels, preds, average="weighted")
        prec, rec = mp.get_precision_recall(labels, preds, average="weighted")
        temp = {"Subset":bias, 
                "Acc":acc, 
                "F1_macro":f1, 
                "Prec_macro":prec, 
                "Recall_macro":rec}
                
        df_biasstat = df_biasstat.append(temp, ignore_index=True)

        time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        cf_matrix_path = os.path.join(dest, "{}_cf_matrix_{}.jpg".format(time_string, bias))
        ax = mp.get_confusion_matrix_ax(mp.get_confusion_matrix_df(labels, preds, label_dict))
        ax.figure.savefig(cf_matrix_path)
        plt.clf()

        bias_cf_matrix_string += template.format(bias, cf_matrix_path)


    fg = mp.get_train_val_curve_kfold(fold_info)
    time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    loss_curve_path = os.path.join(dest, "{}_train_val_bias.jpg".format(time_string))
    fg.savefig(loss_curve_path)
    plt.clf()

    html_testing = '''

    <html>
        <head>
            <title>{page_title}</title>
        </head>
        <body>
            <h1>{model_string}</h1>
            <h3>Report generated on: {date_time}</h3>
            <h2>Class distribution per bias category:</h2>
            {label_distr}
            <h2> Per fold macro averaged metrics </h2>
            {per_fold_metrics}
            <h2> Averaged Metric overview </h2>
            <p> The average and standard dev. of all metrics over the cross validation folds. </p>
            {global_perfold}
            <h2> Per class metrics (over all folds) </h2>
            {c_report}
            <h2> Confusion Matrix </h2>
            <p> Confusion matrix of all predictions over all folds </p>
            <img src='{global_cf_matrix}' width="850">
            <h2> Bias analysis: Metrics per Bias subset </h2>
            {bias_df}
            <h2> Confusion Matrices per bias subset </h2>
            {bias_cm}
            <h2> Train/Validation Curve average over all folds </h2>
            <img src='{loss_curve}' width="850">
        </body>
    </html>
    '''.format(
        page_title="Bias in AI Performance Report using Kfold CV - Model: {}".format(model_name),
        model_string="Model: {} ".format(model_name),
        date_time=datetime.now(),
        label_distr=df_distr.to_html(),
        per_fold_metrics=df_pfmet.to_html(),
        global_perfold = df_stats.to_html(),
        c_report=c_report.to_html(),
        global_cf_matrix=cf_matrix_path_all,
        bias_df=df_biasstat.to_html(),
        bias_cm=bias_cf_matrix_string,
        loss_curve=loss_curve_path
    )

    time = datetime.now().time()
    save_path = os.path.join(dest, '{}_kfoldreport_{}_{}_{}.html'.format(model_name, time.hour, time.minute, time.second))
    with open(save_path, 'w') as f:
        f.write(html_testing)