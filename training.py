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


def main():
    ## ENV VARIABLES
    a_s = 0.65
    b_s = 0.65
    k = 4

    ## TRAINING VARIABLES
    total_timesteps = 8e+6
    exploration_fraction = 0.0125
    # total_timesteps = 1e+6
    # exploration_fraction = 0.1

    lr = 1e-5
    learning_starts = 50000

    ### A2C
    #log_interval = 2000
    ### DQN
    #log_interval = 100
    ### PPO
    log_interval = 5

    save_freq = 250000

    ## OTHER VARIABLES
    log_dir = "./tb_log_dir_k_4"
    saving_model = True
    saving_dir = "./model_weights/with_k_4/PPO"
    saving_name = "PPO_3_k_4"
    saving_path = os.path.join(saving_dir, saving_name)
    debug = True


    # ## ENVIRONMENT
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k,
                                                            mode='fixed'
                                                            )

    ## CALLBACK
    custom_callback = TensorboardCallback(verbose=1, dict_env_params={'a_s': a_s,
                                                                    'b_s': b_s,
                                                                    'k': k,
                                                                    'model_save_path': saving_path})
    # Save a checkpoint every X steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=saving_dir,
        name_prefix=saving_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    ## MODEL
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
    #             learning_rate=lr,
    #             learning_starts=learning_starts,
    #             exploration_fraction=exploration_fraction,
    #             )
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    #model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    if debug:
        print(model.policy)

    model.learn(total_timesteps=total_timesteps,
                log_interval=log_interval,
                progress_bar=True,
                callback=[custom_callback, checkpoint_callback]
                )

    if saving_model:
        model.save(saving_path)




if __name__=="__main__":
    _ = setup.setup()
    main()