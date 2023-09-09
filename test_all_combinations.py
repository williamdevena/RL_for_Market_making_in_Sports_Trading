from stable_baselines3 import A2C, DQN, PPO

from testing import testing
from utils import setup


def test_all_combinations(model_name,
                          num_simulations_per_combination,
                          mode,
                          plot_path):
    """
    Runs simulations on all possible combinations of environment parameters using a trained RL
    agent and save plots of the distributions of various risk and performance metrics.
    """
    _ = setup.setup()

    ### RL AGENTS
    if model_name=="A2C":
        model = A2C.load(f"./model_weights/{model_name}")
    elif model_name=="DQN":
        model = DQN.load(f"./model_weights/{model_name}")
    elif model_name=="PPO":
        model = PPO.load(f"./model_weights/{model_name}")
    ### BASELINE MODELS
    elif model_name in ["fixed_02", "fixed_05", "fixed_08", "random"]:
        model = model_name
    else:
        raise ValueError(f"Invalid value {model_name} for variable 'model_name'. Should be either 'A2C', 'DQN' or 'PPO' (for RL agents) or 'fixed_02', 'fixed_05', 'fixed_08', 'random' (for the baseline models).")

    testing.test_rl_agent_all_combinations(model=model,
                                            num_simulations_per_combination=num_simulations_per_combination,
                                            plot_path=plot_path,
                                            mode=mode
                                            )




def main():
    model_name = "PPO" ## possible values: A2C', 'DQN' or 'PPO' (for RL agents) or
                       ## 'fixed_02', 'fixed_05', 'fixed_08', 'random' (for the baseline models)
    plot_path = f"./plots_testing/{model_name}"
    num_simulations_per_combination = 100
    mode = "long" ## if 'short' uses a smaller set of possible combinations (used for test)
    test_all_combinations(model_name=model_name,
                          num_simulations_per_combination=num_simulations_per_combination,
                          mode=mode,
                          plot_path=plot_path)



if __name__=="__main__":
    main()