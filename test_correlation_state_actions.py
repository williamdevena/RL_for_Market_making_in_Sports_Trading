from stable_baselines3 import A2C, DQN, PPO

from testing import testing
from utils import setup


def test_correlations_state_actions(model_name,
                                    mode,
                                    num_simul_per_comb,
                                    plot_path):
    """
    This module runs simulations on all possible combinations of environment parameters with a given trained RL
    agent, and calculates and saves plots of the mean correlation matrix between state variables and actions.
    """
    _ = setup.setup()

    if model_name=="A2C":
        model = A2C.load(f"./model_weights/{model_name}")
    elif model_name=="DQN":
        model = DQN.load(f"./model_weights/{model_name}")
    elif model_name=="PPO":
        model = PPO.load(f"./model_weights/{model_name}")
    else:
        raise ValueError(f"Invalid value {model_name} for variable 'model_name'. Should be either 'A2C', 'DQN' or 'PPO'.")

    testing.test_correlations_state_vars_and_actions(model=model,
                                                     mode=mode,
                                                     num_simul_per_combin=num_simul_per_comb,
                                                     plot_path=plot_path)



def main():
    model_name = "PPO" ## possible values: A2C', 'DQN' or 'PPO' (for RL agents) or
                       ## 'fixed_02', 'fixed_05', 'fixed_08', 'random' (for the baseline models)
    plot_path = f"corr_matr_actions_test/{model_name}"
    num_simul_per_comb = 100
    mode = "long" ## if 'short' uses a smaller set of possible combinations (used for test)

    test_correlations_state_actions(model_name=model_name,
                                    mode=mode,
                                    num_simul_per_comb=num_simul_per_comb,
                                    plot_path=plot_path)



if __name__=="__main__":
    main()