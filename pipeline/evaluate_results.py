import os
import shutil
import datetime

import pipeline.get_data as get_data
from pipeline.get_data import reset_dir

import sklearn

import numpy as np
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
    if numeric: result = np.array(result).astype(int)
    return result

def get_misclass(pred, actual):
    return prob_to_binary(pred) != prob_to_binary(actual)

def print_test_results(results):
    for k in results["metrics"]: print(f"{k}: {results['metrics'][k]:.3f}")

def print_misclass(results, longform=False):
    pred_binary = prob_to_binary(results["prediction"])
    misclass = get_misclass(results["prediction"], results["actual"])
    if not longform: print("Printing only misclassified items:")
    for i, pred in enumerate(results["prediction"]):
        fpath = results["filenames"][i]
        nicename = os.path.join(os.path.basename(os.path.dirname(fpath)), os.path.basename(fpath))
        pred = results["prediction"][i][0]
        if misclass[i] or longform: print(f"{nicename}\t{pred:.2f}\t{'WRONG' if misclass[i] else ''}")

def generate_misclass_files(results, model_label="default"):
    reset_dir(os.path.join(misclass_root, model_label), assert_exists=False)
    labels = [f"actually_{get_data.conditions[i]}_predicted_{get_data.conditions[abs(1-i)]}" for i in range(2)]
    for label in labels: os.mkdir(os.path.join(misclass_root, model_label, label))

    misclass = get_misclass(results["prediction"], results["actual"])
    for i, fname in enumerate(results["filenames"]):
        if misclass[i]: shutil.copy2(fname, os.path.join(misclass_root, model_label, labels[int(results["actual"][i])]))

def training_accuracy_plot(history, to_file=True, model_label="default", ax=None):
    if ax is None: _,ax = plt.subplots()
    ax.set_ylim(0.5, 1)
    ax.plot(history["accuracy"])
    ax.plot(history["val_accuracy"])
    ax.set(title="Accuracy Across Training", ylabel="Accuracy", xlabel="Epoch")
    ax.legend(["Training", "Validation"])

    if to_file: plt.savefig(os.path.join(plot_root, model_label+"_"+training_accuracy_plot_filename))

def training_loss_plot(history, to_file=True, model_label="default", ax=None):
    if ax is None: _,ax = plt.subplots()
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set(title="Loss Across Training", ylabel="Loss", xlabel="Epoch")
    ax.legend(["Training", "Validation"])

    if to_file: plt.savefig(os.path.join(plot_root, model_label+"_"+training_loss_plot_filename))

def confusion_matrix(results, to_file=True, model_label="default", ax=None):
    if ax is None: _,ax = plt.subplots()
    predicted_binary_num = prob_to_binary(results["prediction"], numeric=True)
    cmat = sklearn.metrics.confusion_matrix(results["actual"], predicted_binary_num)
    labels = list(get_data.conditions)
    cmat = pd.DataFrame(cmat, index=labels, columns=labels)
    # print(cmat)
    
    sns.heatmap(data=cmat, annot=True, fmt="d", cbar=None, cmap="Blues", ax=ax)
    ax.set(title="Confusion Matrix", ylabel="Actual", xlabel="Predicted")

    if to_file: plt.savefig(os.path.join(plot_root, model_label+"_"+confusion_matrix_plot_filename))

def combine_history(histories):
    history = {}
    known_keys = {"accuracy", "val_accuracy", "loss", "val_loss"}  # Only average things we know to be averageable
    for k in known_keys:
        if k in histories[0]: history[k] = np.mean([h[k] for h in histories], axis=0)
    return history

def combine_results(resultses):
    metrics = {}
    known_metrics = {"loss", "accuracy"}
    known_toplevel_to_append = {"prediction", "actual", "filenames"}
    for k in known_metrics:
        if k in resultses[0]["metrics"]: metrics[k] = np.mean([h["metrics"][k] for h in resultses])
    combined_results = {"metrics": metrics}
    for k in known_toplevel_to_append:
        combined_results[k] = np.concatenate([np.array(r[k]) for r in resultses])
        # combined_results[k] = list(itertools.chain.from_iterable([r[k] for r in resultses]))
    return combined_results

def format_datetime(datetime):
    return f"{datetime:%I:%M:%S%p}"

def format_time_estimate(duration=None):
    status = f"Starting at {format_datetime(datetime.datetime.now())}"
    if duration is not None:
        est_done = datetime.datetime.now()+datetime.timedelta(seconds=duration)
        status += f"; estimated time {duration:.2f}s to finish at {format_datetime(est_done)}"
    return status+"…"

def format_finished_msg(duration):
    return f"Finished in {duration:.2f}s at {format_datetime(datetime.datetime.now())}."
