import os
import shutil

import get_data
from get_data import reset_dir

import sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

evaluation_metrics = ["accuracy"]
binary_threshold = 0.5

misclass_root = os.path.abspath("../models/misclassified")
plot_root = os.path.abspath("../models/plots")
training_accuracy_plot_filename = "accuracy_plot.png"
training_loss_plot_filename = "loss_plot.png"
confusion_matrix_plot_filename = "confusion_matrix.png"

# This function must be used whenever we expect a binary yes/no answer, rather than a probability!
# We want to only do this thresholding in one place.
def prob_to_binary(ds, threshold=None, numeric=False):
    if threshold is None: threshold = binary_threshold
    result = ds > threshold
    if numeric: result = result.numpy().astype(int)
    return result

def get_misclass(pred, actual):
    return prob_to_binary(pred) != prob_to_binary(actual)

def print_test_results(results, longform=False):
    for k in results["metrics"]: print(f"{k}: {results['metrics'][k]}")
    pred_binary = prob_to_binary(results["prediction"])
    misclass = get_misclass(results["prediction"], results["actual"])
    if not longform: print("Printing only misclassified items:")
    for i, pred in enumerate(results["prediction"]):
        fpath = results["filenames"][i]
        nicename = os.path.join(os.path.basename(os.path.dirname(fpath)), os.path.basename(fpath))
        pred = results["prediction"][i][0]
        if misclass[i] or longform: print(f"{nicename}\t{pred:.2f}\t{'WRONG' if misclass[i] else ''}")

def generate_misclass_files(results):
    reset_dir(misclass_root)
    labels = [f"actually_{get_data.conditions[i]}_predicted_{get_data.conditions[abs(1-i)]}" for i in range(2)]
    for label in labels: os.mkdir(os.path.join(misclass_root, label))

    misclass = get_misclass(results["prediction"], results["actual"])
    for i, fname in enumerate(results["filenames"]):
        if misclass[i]: shutil.copy2(fname, os.path.join(misclass_root, labels[int(results["actual"][i])]))

def training_accuracy_plot(history, to_file=True):
    plt.figure()
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Accuracy Across Training")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"])

    if to_file: plt.savefig(os.path.join(plot_root, training_accuracy_plot_filename))

def training_loss_plot(history, to_file=True):
    plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss Across Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"])

    if to_file: plt.savefig(os.path.join(plot_root, training_loss_plot_filename))

def confusion_matrix(results, to_file=True):
    predicted_binary_num = prob_to_binary(results["prediction"], numeric=True)
    cmat = sklearn.metrics.confusion_matrix(results["actual"], predicted_binary_num)
    labels = list(reversed(get_data.conditions))  # sklearn interprets categories reversedly, it seems
    cmat = pd.DataFrame(cmat, index=labels, columns=labels)
    # print(cmat)
    
    plt.figure()
    sns.heatmap(data=cmat, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if to_file: plt.savefig(os.path.join(plot_root, confusion_matrix_plot_filename))
