
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




    # ## EXTRACT ORDER INTENSITIES AND VOLUME RATES FOR LOB SIMULATION
    # price_file_path = os.path.join(data_directory, "djokovic_match_odds/29/32060431/1.209205488.bz2")
    # id_runner = 2249229

    # inplay_idx = pricefileutils.get_idx_first_inplay_mb_from_prices_file(price_file_path)
    # market_books = betfairutil.read_prices_file(price_file_path)
    # inplay_mbs = market_books[inplay_idx:]

    # dict_rates_back, dict_rates_lay = data_processing.extract_orders_rates_for_lob_simulation(market_books=inplay_mbs,
    #                                                         id_runner=id_runner)

    # for name_rate, rate in dict_rates_back.items():
    #     print(f"{name_rate}: {rate}\n")

    # for name_rate, rate in dict_rates_lay.items():
    #     print(f"{name_rate}: {rate}\n")








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
    # start_time = time.time()


    # simulator_framework.run_single_simulation(price_simulator=price_simulator,
    #                                    strategy=strategy,
    #                                    num_simulations=num_simulations,
    #                                    plotting=True,
    #                                    plot_path=plot_path)
    # print("--- %s seconds ---" % (time.time() - start_time))








    # ## TEST FIXED STRATEGY
    # a_s = 0.7
    # b_s = 0.7

    # price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)
    # simulator_framework = AvellanedaStoikovFramework()
    # num_simulations = 100

    # for offset in np.arange(0.1, 1.1, 0.1):
    #     print(f"\nOFFSET: {offset}")
    #     strategy = FixedOffsetStrategy(offset=offset)
    #     simulator_framework.run_simulation(price_simulator=price_simulator,
    #                                        strategy=strategy,
    #                                        num_simulations=num_simulations,
    #                                        plotting=False)







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








    ### TESTING GYM ENVIRONMENT
    a_s = 0.7
    b_s = 0.7
    k = 7
    env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s, b_s=b_s, k=k)

    prices = []
    back_prices = []
    lay_prices = []
    pnl_list = []
    inventory = []
    done = False
    #for x in range(200):
    while not done:
        # rb = env.price[env.timestep] + 0.2
        # rl = env.price[env.timestep] - 0.2
        #action = (rb, rl)

        action = env.action_space.sample()
        rb = env.price[env.timestep] + action[0]
        rl = env.price[env.timestep] - action[1]

        back_prices.append(rb)
        lay_prices.append(rl)

        [price, _], pnl, done = env.step(action=action)
        pnl_list.append(pnl)
        prices.append(price)
        inventory.append(env.q['stake'])

    plt.plot(prices)
    plt.plot(back_prices)
    plt.plot(lay_prices)

    #plt.plot(pnl_list)
    #plt.plot(inventory)

    plt.show()








if __name__=="__main__":
    main()