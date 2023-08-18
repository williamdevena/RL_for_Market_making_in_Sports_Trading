
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
from src import data_processing, plotting
from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy
from strategy.fixedOffsetStrategy import FixedOffsetStrategy
from strategy.randomStrategy import RandomStrategy
from utils import pricefileutils, setup

from . import testing

#import tennisim



def main():
    data_directory = setup.setup()

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







    # #### TESTING CASH OUT MECHANISM (PNL CALCULATION)
    # framework = AvellanedaStoikovFramework()
    # cashout = framework.calculate_cash_out(stake=-33.3, odds=-1.1, current_odds=1.01)
    # print(cashout)








    # ### TESTING GYM ENVIRONMENT
    # a_s = 0.7
    # b_s = 0.7
    # k = 10
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s, b_s=b_s, k=k)

    # # prices = []
    # # back_prices = []
    # # lay_prices = []
    # # pnl_list = []
    # # inventory = []

    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     _, _, done, _, _ = env.step(action=action)

    # plt.plot(env.price, label="Mid-price")
    # plt.plot(env.list_momentum_indicator, label="Momentum indicator")
    # # plt.plot(env.back_prices)
    # # plt.plot(env.lay_prices)
    # plt.legend()
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

    # # It will check the custom environment and output additional warnings if needed
    # check_env(env)






    ### TEST STABLE-BASELINES (CART POLE)
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













if __name__=="__main__":
    main()