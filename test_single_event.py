"""
This module executes a single event simulation with fixed environment parameters using
a given trained RL agent, and plots relevant graphs.
"""

from stable_baselines3 import A2C, DQN, PPO

from environments.gym_env import sportsTradingEnvironment
from testing import testing
from utils import setup


def main():
    _ = setup.setup()

    ### TEST RL ON SINGLE EPISODE
    a_s = 0.65
    b_s = 0.65
    k = 4
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k,
                                                            mode='fixed')

    #model = DQN.load("./model_weights/DQN)
    #model = PPO.load("./model_weights/PPO")
    model = A2C.load("./model_weights/A2C")


    #for x in range(10):
    testing.test_rl_agent_single_episode(model=model, env=env, debug=False, plot_results=True)



if __name__=="__main__":
    main()
