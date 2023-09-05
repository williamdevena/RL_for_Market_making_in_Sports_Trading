"""
This module executes the training of the RL agents.
"""

import os

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from environments.gym_env import sportsTradingEnvironment
from environments.gym_env.tensorboardCallback import TensorboardCallback
from utils import setup


def training(model_name, saving_model, debug, saving_dir, saving_name, log_dir):
    hyperparams = setup.setup_training_hyperparameters()

    if model_name not in {"DQN", "PPO", "A2C"}:
        raise ValueError(f"Invalid value {model_name} for 'model_name' variable, it must be either \"DQN\", \"PPO\" or \"A2C\".")

    dict_hyperparams = hyperparams[model_name]

    ## OTHER VARIABLES
    # log_dir = "./tb_log_dir_k_4"
    # saving_model = True
    # saving_dir = f"./model_weights/{model_name}"
    # saving_name = model_name + "_1"
    saving_path = os.path.join(saving_dir, saving_name)
    # debug = False


    # ## ENVIRONMENT
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=dict_hyperparams['a_s'],
                                                            b_s=dict_hyperparams['b_s'],
                                                            k=dict_hyperparams['k'],
                                                            mode='fixed'
                                                            )

    ## CALLBACK
    custom_callback = TensorboardCallback(verbose=1, dict_env_params={'a_s': dict_hyperparams['a_s'],
                                                                    'b_s': dict_hyperparams['b_s'],
                                                                    'k': dict_hyperparams['k'],
                                                                    'model_save_path': saving_path})
    # Save a checkpoint every X steps
    checkpoint_callback = CheckpointCallback(
        save_freq=dict_hyperparams['save_freq'],
        save_path=saving_dir,
        name_prefix=saving_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    ## MODEL
    if model_name=="DQN":
        model = DQN("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=dict_hyperparams['lr'],
                    learning_starts=dict_hyperparams['learning_starts'],
                    exploration_fraction=dict_hyperparams['exploration_fraction'],
                    )
    elif model_name=="PPO":
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=dict_hyperparams['lr'],
                    learning_starts=dict_hyperparams['learning_start'],
                    exploration_fraction=dict_hyperparams['exploration_fraction'],
                    )
    elif model_name=="A2C":
        model = A2C("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=dict_hyperparams['lr'],
                    learning_starts=dict_hyperparams['learning_start'],
                    exploration_fraction=dict_hyperparams['exploration_fraction'],)
    else:
        raise ValueError(f"Invalid value {model_name} for 'model_name' variable, it must be either \"DQN\", \"PPO\" or \"A2C\".")

    if debug:
        print(model.policy)

    model.learn(total_timesteps=dict_hyperparams['total_timesteps'],
                log_interval=dict_hyperparams['log_interval'],
                progress_bar=True,
                callback=[custom_callback, checkpoint_callback]
                )

    if saving_model:
        model.save(saving_path)




def main():
    model_name = "DQN"
    saving_model = True
    debug = True
    saving_dir = "./model_weights_test"
    saving_name = model_name
    log_dir = "tb_log_dir_new_test"

    training(model_name=model_name,
             saving_model=saving_model,
             debug=debug,
             saving_dir=saving_dir,
             saving_name=saving_name,
             log_dir=log_dir)




if __name__=="__main__":
    _ = setup.setup()
    main()