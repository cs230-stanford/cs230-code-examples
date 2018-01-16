"""Peform hyperparemeters search
"""

import os
import argparse
from subprocess import check_call

import numpy as np

from model.utils import Params


PYTHON = "python3"
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/test')


def launch_training_job(parent_dir, job_name, params):
    """
    Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        params: (dict) containing hyperparameters
    """
    # create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    # write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)


    # launch training with this config
    cmd = "{python} train.py --model_dir={model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # perform hypersearch over one parameter
    learning_rates = [1e-2, 1e-3, 1e-4]

    for learning_rate in learning_rates:
        # modify the relevant parameter in params
        params.learning_rate = learning_rate

        # launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, job_name, params)
