"""
This module runs simulations on all possible combinations of environment parameters with a given trained RL
agent, and calculates and saves plots of the mean correlation matrix between state variables and actions.
"""

from stable_baselines3 import A2C, DQN, PPO

from testing import testing
from utils import setup


def main():
    _ = setup.setup()

    # model_name = "PPO_2_k_4"
    # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # model_name = "PPO_3_k_4"
    # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # model_name = "DQN_2_k_4"
    # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    model_name = "A2C_2_k_4"
    model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")


    plot_path = f"corr_matr_actions_test/{model_name}"
    num_simul_per_comb = 100
    mode = "long" ## if 'short' uses a smaller set of possible combinations (used for test)
    testing.test_correlations_state_vars_and_actions(model=model,
                                                     mode=mode,
                                                     num_simul_per_combin=num_simul_per_comb,
                                                     plot_path=plot_path)

if __name__=="__main__":
    main()