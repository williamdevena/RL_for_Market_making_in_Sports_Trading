"""
This module runs simulations on all possible combinations of environment parameters using a trained RL
agent and save plots of the distributions of various risk and performance metrics.
"""

from stable_baselines3 import A2C, DQN, PPO

from testing import testing
from utils import setup


def main():
    _ = setup.setup()

    # model_name = "fixed_02"
    # model = model_name

    # model_name = "fixed_05"
    # model = model_name

    # model_name = "fixed_08"
    # model = model_name

    # model_name = "random"
    # model = model_name

    # model_name = "PPO_1_k_4"
    # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # model_name = "A2C_1_k_4"
    # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    # model_name = "DQN_1_k_4"
    # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # model_name = "A2C_2_k_4"
    # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    # model_name = "DQN_2_k_4"
    # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # model_name = "PPO_2_k_4"
    # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    model_name = "PPO_3_k_4"
    model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")



    plot_path = f"./plots_test_2/{model_name}"
    num_simulations_per_combination = 1
    mode = "long" ## if 'short' uses a smaller set of possible combinations (used for test)
    testing.test_rl_agent_all_combinations(model=model,
                                   num_simulations_per_combination=num_simulations_per_combination,
                                   plot_path=plot_path,
                                   mode=mode
                                   )


if __name__=="__main__":
    main()