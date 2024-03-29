# Some more results evaluation stuff, put here to avoid a circular dependency between evaluate_results and learner

import matplotlib.pyplot as plt
import time

import evaluate_results
from evaluate_results import format_time_estimate, format_finished_msg
import learner

import tensorflow as tf

def hyperparameters_to_string(model_name, n_epochs, batch_size, steps_per_epoch):
    params = {"model": model_name, "epochs": n_epochs, "batch-size": batch_size, "steps-per-epoch": steps_per_epoch}
    return "_".join([f"{k}={params[k]}" for k in params])

def evaluation_figs(history, results, label):
    fig,axs = plt.subplots(1, 3, figsize=(20, 6))
    evaluate_results.training_accuracy_plot(history, model_label=label, ax=axs[0])
    evaluate_results.training_loss_plot(history, model_label=label, ax=axs[1])
    evaluate_results.confusion_matrix(results, model_label=label, ax=axs[2])
    fig.suptitle(label)

def evaluate_hyperparameters(model_fn, data, n_epochs, batch_size, steps_per_epoch, n_samples=1, print_samples=False, verbose=0):
    train_ds, valid_ds, test_ds = data
    input_shape = train_ds.element_spec[0].shape[1:]
    label = hyperparameters_to_string(model_fn.__name__.replace("_", "-"), n_epochs, batch_size, steps_per_epoch)
    history, results = [], []
    if batch_size is not None:
        this_test_ds = test_ds.unbatch().batch(batch_size)
        this_test_ds.file_paths = test_ds.file_paths
    else: this_test_ds = test_ds

    n_batches = tf.data.experimental.cardinality(this_test_ds).numpy()
    t_per_batch = None

    t0_overall = time.time()
    for i in range(n_samples):
        model = model_fn(input_shape)  # Recreate the model each time for independent samples

        if print_samples:
            status = f"{label} ROUND {i}/{n_samples}. "
            status += format_time_estimate(None if t_per_batch is None else t_per_batch*n_batches)
            print(status)
        t0 = time.time()

        this_history = learner.train(model, n_epochs, train_ds.repeat(), valid_ds,
            steps_per_epoch=steps_per_epoch, verbose=verbose).history
        history.append(this_history)
        this_results = learner.test(model, this_test_ds, verbose=verbose)
        results.append(this_results)
        
        t1 = time.time()
        if t_per_batch is None: t_per_batch = (t1-t0)/n_batches
        else: t_per_batch = (t_per_batch+(t1-t0)/n_batches)/2
        if print_samples: print(format_finished_msg(t1-t0))
    t1_overall = time.time()

    history = evaluate_results.combine_history(history)
    results = evaluate_results.combine_results(results)
    print(f"{label} ({(t1_overall-t0_overall):.2f}s)")
    evaluate_results.print_test_results(results)
    evaluation_figs(history, results, label)
    evaluate_results.generate_misclass_files(results, model_label=label)
    return history, results
