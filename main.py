
import os
import pickle
import random
import time
from pprint import pprint

import betfairutil
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tennisim
from alive_progress import alive_it

from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
from environments.gym_env import sportsTradingEnvironment
#from environments.tennis_markov import tennisMarkovSimulator
from environments.tennis_simulator import tennisSimulator
from src import data_processing, plotting, testing
from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy
from strategy.fixedOffsetStrategy import FixedOffsetStrategy
from strategy.randomStrategy import RandomStrategy
from utils import pricefileutils

#import tennisim



def main():
    dotenv.load_dotenv()
    data_directory = os.environ.get("DATA_DIRECTORY")
    random.seed(42)


    # ## TENNIS MARKOV SIMULATOR
    # s = 0.5
    # t = 0.5
    # simulator = tennisMarkovSimulator.TennisMarkovSimulator(s=s, t=t)
    # num_simulations = 100

    # for x in range(num_simulations):
    #     prob_list, odds_list, games_idx = simulator.simulate()

    #     plt.plot(prob_list)
    #     # plt.plot(odds_list)

    #     simulator.restart()
    # plt.ylim(-0.1, 1.1)
    # plt.show()







    #### AVELLANEDA-STOIKOV WITH TENNIS SIMULATOR
    # s = 0.7
    # t = 0.3
    # price_simulator = tennisMarkovSimulator.TennisMarkovSimulator(s=s, t=t)
    # simulator_framework = AvellanedaStoikovFramework()

    # strategy = RandomStrategy(range_offset=(0, 1))
    # #strategy = FixedOffsetStrategy(offset=0.2)
    # #strategy = AvellanedaStoikovStrategy()
    # simulator_framework.run_simulation(price_simulator=price_simulator, strategy=strategy, num_simulations=1000)








    # #### TENNISIM TEST
    # import tennisim
    # from tennisim import match
    # from tennisim.sim import sim_game, sim_match, sim_set

    # a_s = 0.7
    # b_s = 0.7
    # # won, scores = sim_game(p)
    # # print(scores)

    # # won, scores1, scores2 = sim_set(p, b_s=0.5)
    # # print(len(scores2))

    # #won, match_scores, set_scores, game_scores = sim_match(a_s=a_s, b_s=b_s)

    # # print(match_scores)
    # # print(set_scores)
    # # print(game_scores)

    # num_simulations = 100

    # for x in range(num_simulations):
    #     result = match.reformat_match(match_data=sim_match(a_s=a_s, b_s=b_s), p_a=a_s, p_b=b_s)
    #     list_probs = [x['prob'] for x in result]
    #     list_probs_w = [x['prob_w'] for x in result]
    #     list_probs_l = [x['prob_l'] for x in result]
    #     plt.plot(list_probs)

    # plt.show()








    # ### AS SIMULATION USING TENNISIM
    # a_s = 0.65
    # b_s = 0.65
    # k = 2

    # price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)

    # simulator_framework = AvellanedaStoikovFramework(k=k)

    # #strategy = RandomStrategy(range_offset=(0, 1))
    # #plot_path = "./plots/random"

    # strategy = FixedOffsetStrategy(offset=0.2)
    # plot_path = "./plots_single_simulations/fixed_02"

    # # strategy = AvellanedaStoikovStrategy()
    # # plot_path = "./plots_single_simulations/as"

    # num_simulations = 10

    # simulator_framework.run_single_simulation(price_simulator=price_simulator,
    #                                    strategy=strategy,
    #                                    num_simulations=num_simulations,
    #                                    plotting=True,
    #                                    plot_path=plot_path)








    ### TESTING STRATEGIES
    # strategy = FixedOffsetStrategy(offset=0.2)
    # plot_path = "./plots/fixed_02"

    # strategy = FixedOffsetStrategy(offset=0.5)
    # plot_path = "./plots/fixed_05"

    # strategy = FixedOffsetStrategy(offset=0.8)
    # plot_path = "./plots/fixed_08"

    # strategy = RandomStrategy(range_offset=(0, 1))
    # plot_path = "./plots/random"

    # num_simulations_per_combination = 100
    # testing.test_strategies(plot_path=plot_path,
    #                         strategy=strategy,
    #                         num_simulations_per_combination=num_simulations_per_combination)










    # #### PLOT FINAL PNL DISTRIBUTIONS OF MODELS IN SAME GRAPH
    # strategy_names = [ "fixed_02",
    #                   "random", "fixed_05",
    #                   "fixed_08"
    #                   ]
    # metrics = [
    #     #"final_pnl",
    #     # "volatility",
    #     "mean_return",
    #     #"min_pnl",
    #     # "max_pnl",
    #     # "sharpe_ratio",
    #     # "sortino_ratio",
    #     # "mean_inv_stake"
    #     ]

    # plotting.plot_results_of_all_strategies_test(results_path="./plots",
    #                                              strategies_names_list=strategy_names,
    #                                              metrics_list=metrics)







    #### TESTING CAH OUT MECHANISM (PNL CALCULATION)
    framework = AvellanedaStoikovFramework()
    cashout = framework.calculate_cash_out(stake=-33.3, odds=-1.1, current_odds=1.01)
    print(cashout)








    # ### TESTING GYM ENVIRONMENT
    # a_s = 0.7
    # b_s = 0.7
    # k = 7
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s, b_s=b_s, k=k)

    # # prices = []
    # # back_prices = []
    # # lay_prices = []
    # pnl_list = []
    # inventory = []
    # done = False
    # #for x in range(200):
    # while not done:
    #     action = env.action_space.sample()

    #     [_, _], pnl, done = env.step(action=action)
    #     pnl_list.append(pnl)
    #     #prices.append(price)
    #     inventory.append(env.q['stake'])

    # plt.plot(env.price)
    # plt.plot(env.back_prices)
    # plt.plot(env.lay_prices)
    # plt.show()
    # plt.close()

    # plt.plot(pnl_list, label="PnL")
    # plt.plot(inventory, label="Inventory")
    # plt.legend()
    # plt.show()





    # ### TEST ENVRIONMENT FOR ERRORS
    # from stable_baselines3.common.env_checker import check_env
    # from stable_baselines3.common.env_util import make_vec_env

    # # Instantiate the env
    # a_s = 0.65
    # b_s = 0.65
    # k = 2

    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k)
    # # It will check your custom environment and output additional warnings if needed
    # check_env(env)






    ### TEST STABLE-BASELINES

    ### EXAMPLE TRAIN ON CARTPOLE
    # import gymnasium as gym
    # from stable_baselines3 import DQN

    # env = gym.make("CartPole-v1")
    # log_dir = "./cartpole_dir"
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # model.learn(total_timesteps=100000,
    #             log_interval=10,
    #             progress_bar=True,
    #             #callback=callback
    #             )






    # ### TRAIN
    # import gymnasium as gym
    # from stable_baselines3 import DQN, PPO

    # from environments.gym_env.tensorboardCallback import TensorboardCallback

    # # # env = gym.make("CartPole-v1", render_mode="human")
    # a_s = 0.65
    # b_s = 0.65
    # k = 10
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k)

    # #callback = TensorboardCallback(verbose=1)
    # #print(dir(TensorboardCallback))

    # log_dir = "./test_log_dir"
    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
    #             #learning_starts=1000
    #             )

    # print(model.replay_buffer.__dict__.keys())
    # print(model.replay_buffer.observations)
    # print(model.replay_buffer.buffer_size)
    # print(model.replay_buffer.obs_shape)

    # log_dir = "./test_log_dir"
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # model.learn(total_timesteps=1000000,
    #             log_interval=100,
    #             progress_bar=True,
    #             #callback=callback
    #             )

    # print(callback.training_env)


    # model.save("test_DQN")


    ## TEST
    # del model # remove to demonstrate saving and loading
    # model = DQN.load("test_DQN")

    # #print(env.reset())
    # obs, info = env.reset()
    # #prices = []
    # terminated = False

    # while not terminated:
    #     #print(obs)
    #     action, _states = model.predict(obs, deterministic=True)
    #     #print(action)
    #     obs, reward, terminated, truncated, info = env.step(action)

    # # plt.plot(env.price, label="mid-price")
    # # plt.plot(env.back_prices, label="back price")
    # # plt.plot(env.lay_prices, label="lay price")
    # # plt.legend()
    # # print(len(env.price), len(env.back_prices), len(env.lay_prices))

    # plt.plot(env.price, label="mid-price")
    # plt.plot(env.list_pnl, label="PnL")
    # plt.plot(env.list_inventory_stake, label="Stake")
    # plt.plot(env.list_inventory_odds, label="Odds")
    # plt.legend()

    # plt.show()




    # #### TEST MODEL ON ENV
    # import gymnasium as gym
    # from stable_baselines3 import DQN

    # # # env = gym.make("CartPole-v1", render_mode="human")
    # a_s = 0.65
    # b_s = 0.65
    # k = 7
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k)

    # #model = DQN.load("test_DQN")
    # model = DQN("MlpPolicy", env, verbose=1)

    # num_episodes = 2
    # list_final_pnl = []
    # for episode in alive_it(range(num_episodes)):
    #     obs, info = env.reset()
    #     terminated = False
    #     while not terminated:
    #         ## MODEL
    #         action, _states = model.predict(obs, deterministic=True)
    #         ## RANDOM
    #         # action = env.action_space.sample()

    #         print(obs, action)

    #         #print(action)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #     list_final_pnl.append(env.list_pnl[-1])

    #     f = plt.figure(figsize=(10, 10))
    #     f.add_subplot(2, 2, 1)
    #     plt.plot(env.price, label="mid-price")
    #     plt.plot(env.back_prices, label="mid-price")
    #     plt.plot(env.lay_prices, label="mid-price")
    #     plt.legend()
    #     f.add_subplot(2, 2, 2)
    #     plt.plot(env.back_offsets, label="back offset")
    #     plt.plot(env.lay_offsets, label="lay offset")
    #     plt.legend()
    #     f.add_subplot(2, 2, 3)
    #     plt.plot(env.list_pnl, label="PnL")
    #     plt.legend()
    #     f.add_subplot(2, 2, 4)
    #     plt.plot(env.list_inventory_stake, label="Stake")
    #     plt.plot(env.list_inventory_odds, label="Odds")
    #     plt.legend()

    # # mean_pnl = np.mean(list_final_pnl)
    # # print(f"Mean pnl: {mean_pnl}")

    # # plt.plot(list_final_pnl)
    # plt.show()






if __name__=="__main__":
    main()