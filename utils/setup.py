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
    """
    Main setup function. Calls every other setup function needed.
    Should be called at the beggining of each execution.
    """
    dotenv.load_dotenv()
    data_directory = os.environ.get("DATA_DIRECTORY")

    set_random_seeds()

    return data_directory


def set_random_seeds():
    """
    Sets the random seeds for reproducibility.
    """
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)


def setup_training_hyperparameters():
    """
    Defines and returns the specified hyperparamters for the agent's training.

    Returns:
        dict_hyperparameters (dict): contains the hyperparaemters for each type
            of agent.
    """
    dict_hyperparameters = {
        "DQN": {'a_s': 0.65,
                'b_s': 0.65,
                'k': 4,
                'total_timesteps': 4e+6,
                'exploration_fraction': 0.025,
                'lr': 0.00001,
                'learning_starts': 50000,
                'log_interval': 100,
                'save_freq': 250000,
                },
        "PPO": {'a_s': 0.65,
                'b_s': 0.65,
                'k': 4,
                'total_timesteps': 4e+6,
                'exploration_fraction': 0.025,
                'lr': 0.0003,
                'learning_starts': 50000,
                'log_interval': 5,
                'save_freq': 250000,
                },
        "A2C": {'a_s': 0.65,
                'b_s': 0.65,
                'k': 4,
                'total_timesteps': 4e+6,
                'exploration_fraction': 0.025,
                'lr': 0.0007,
                'learning_starts': 50000,
                'log_interval': 2000,
                'save_freq': 250000,
                }
    }

    return dict_hyperparameters