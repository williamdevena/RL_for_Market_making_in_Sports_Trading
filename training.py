import gymnasium as gym
from stable_baselines3 import DQN, PPO

from environments.gym_env import sportsTradingEnvironment
from environments.gym_env.tensorboardCallback import TensorboardCallback
from utils import setup


def main():
    ## VARIABLES
    a_s = 0.65
    b_s = 0.65
    k = 10
    log_dir = "./test_log_dir"
    lr = 1e-5
    learning_starts=50000
    exploration_fraction = 0.2
    saving_model = True
    saving_path = "test_DQN"
    debug = False


    ## ENVIRONMENT
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k)
    ## CALLBACK
    #callback = TensorboardCallback(verbose=1)
    #print(dir(TensorboardCallback))

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

    model.learn(total_timesteps=1000000,
                log_interval=100,
                progress_bar=True,
                #callback=callback
                )

    if saving_model:
        model.save(saving_path)




if __name__=="__main__":
    _ = setup.setup()
    main()