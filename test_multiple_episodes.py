from stable_baselines3 import A2C, DQN, PPO

from environments.gym_env import sportsTradingEnvironment
from testing import testing
from utils import setup


def main():
    _ = setup.setup()

    a_s = 0.65
    b_s = 0.65
    k = 4
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                            b_s=b_s,
                                                            k=k,
                                                            mode='fixed')


    # # model = "fixed_02"
    # # plot_path = f"plots_k_4/{model}"

    # # model = "fixed_05"
    # # plot_path = f"plots_k_4/{model}"

    # # model = "fixed_08"
    # # plot_path = f"plots_k_4/{model}"

    # # model = "random"
    # # plot_path = f"plots_k_4/{model}"


    # # model_name = "PPO_2_k_4"
    # # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # # model_name = "DQN_1_k_4"
    # # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # # model_name = "DQN_2_k_4"
    # # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # # model_name = "A2C_1_k_4"
    # # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    # model_name = "A2C_2_k_4"
    # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    model_name = "PPO_3_k_4"
    model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")



    plot_path = f"plots_test_/{model_name}"
    num_episodes = 100
    testing.test_rl_agent_multiple_episods(num_episodes=num_episodes,
                                            model=model,
                                            env=env,
                                            plot_results=True,
                                            plot_path=plot_path,
                                            debug=False)


if __name__=="__main__":
    main()
