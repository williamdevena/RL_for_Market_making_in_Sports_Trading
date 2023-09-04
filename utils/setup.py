"""
This module contains functions used to setup useful environment variables and the random seeds, at the beginning of each execution
to ensure reproducibility.
"""

import os
import random

import dotenv
import numpy as np
import tensorflow as tf
import torch


def setup():
    dotenv.load_dotenv()
    data_directory = os.environ.get("DATA_DIRECTORY")

    set_random_seeds()

    return data_directory


def set_random_seeds():
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
