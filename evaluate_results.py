import os
import shutil

import get_data
from get_data import reset_dir

evaluation_metrics = ["accuracy"]
binary_threshold = 0.5

misclass_root = os.path.abspath("../models/misclassified")

def prob_to_binary(ds, threshold=None):
    if threshold is None: threshold = binary_threshold
    return ds > threshold

def get_misclass(pred, actual):
    return prob_to_binary(pred) != prob_to_binary(actual)

def print_test_results(results, longform=False):
    for k in results["history"]: print(f"{k}: {results['history'][k]}")
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
