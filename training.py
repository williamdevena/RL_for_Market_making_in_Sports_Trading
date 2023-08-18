import os

import gymnasium as gym
from stable_baselines3 import DQN, PPO

from environments.gym_env import sportsTradingEnvironment
from environments.gym_env.tensorboardCallback import TensorboardCallback
from utils import setup


def main():
    ## ENV VARIABLES
    a_s = 0.65
    b_s = 0.65
    k = 7

    ## TRAINING VARIABLES
    total_timesteps = 1e+6
    lr = 1e-5
    learning_starts=50000
    exploration_fraction = 0.2
    log_interval = 100

    ## OTHER VARIABLES
    log_dir = "./test_log_dir"
    saving_model = True
    saving_dir = "./model_weights"
    saving_name = "DQN_1"
    saving_path = os.path.join(saving_dir, saving_name)
    debug = False


    ## ENVIRONMENT
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k)
    ## CALLBACK
    callback = TensorboardCallback(verbose=1)

    ## MODEL
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=lr,
                learning_starts=learning_starts,
                exploration_fraction=exploration_fraction,
                )
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    if debug:
        print(model.replay_buffer.__dict__.keys())
        print(model.replay_buffer.observations)
        print(model.replay_buffer.buffer_size)
        print(model.replay_buffer.obs_shape)

    model.learn(total_timesteps=total_timesteps,
                log_interval=log_interval,
                progress_bar=True,
                callback=callback
                )

    if saving_model:
        model.save(saving_path)




if __name__=="__main__":
    _ = setup.setup()
    main()