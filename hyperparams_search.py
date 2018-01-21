"""Peform hyperparemeters search
"""

import argparse
import json
import os
from subprocess import check_call

import numpy as np
from tabulate import tabulate

from model.utils import Params


PYTHON = "python3"
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate')


def launch_training_job(parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


def synthesize_metrics(parent_dir, save_file):
    """Synthesize the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: parent directory of the different experiments launched
        save_file: where to save a table of the results (markdown format)
    """
    metrics = dict()

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        # Get the metrics for this experiment
        metrics_file = os.path.join(parent_dir, subdir, 'metrics_eval_best_weights.json')
        if os.path.isfile(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics[subdir] = json.load(f)
        else:
            print("Couldn't find any metrics json file at {}".format(metrics_file))

    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    print("Results of experiments in {}".format(parent_dir))
    print(res)

    # Save results in save_file
    with open(save_file, 'w') as f:
        f.write(res)

    return res


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [1e-4, 1e-3, 1e-2]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, job_name, params)

    # Synthesize metrics into parent_dir/results.md
    synthesize_metrics(args.parent_dir, os.path.join(args.parent_dir, "results.md"))
