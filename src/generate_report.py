from sklearn.metrics import classification_report
import metrics_and_plotting as mp
from configs.paths import report_dir

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
