import os

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

from environments.gym_env import sportsTradingEnvironment
from environments.gym_env.tensorboardCallback import TensorboardCallback
from utils import setup


def main():
    ## ENV VARIABLES
    a_s = 0.65
    b_s = 0.65
    k = 7

    ## TRAINING VARIABLES
    total_timesteps = 2e+6
    lr = 1e-6
    learning_starts = 50000
    exploration_fraction = 0.1
    log_interval = 100

    ## OTHER VARIABLES
    log_dir = "./tb_log_dir"
    saving_model = True
    saving_dir = "./model_weights"
    #saving_name = "DQN_3"
    saving_name = "A2C_2"
    saving_path = os.path.join(saving_dir, saving_name)
    debug = False


    # ## ENVIRONMENT
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k)
    # Parallel environments
    #vec_env = make_vec_env("CartPole-v1", n_envs=4)



    ## CALLBACK
    callback = TensorboardCallback(verbose=1, dict_env_params={'a_s': a_s,
                                                                'b_s': b_s,
                                                                'k': k})

    ## MODEL
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
    #             learning_rate=lr,
    #             learning_starts=learning_starts,
    #             exploration_fraction=exploration_fraction,
    #             )
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    model = A2C("MlpPolicy",
                #vec_env,
                env,
                verbose=1, tensorboard_log=log_dir,
                # learning_rate=lr,
                # learning_starts=learning_starts,
                # exploration_fraction=exploration_fraction
                )

    if debug:
        # print(model.replay_buffer.__dict__.keys())
        # print(model.replay_buffer.observations)
        # print(model.replay_buffer.buffer_size)
        # print(model.replay_buffer.obs_shape)
        print(model.policy)

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