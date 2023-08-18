import os
import random

import dotenv
import numpy as np
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