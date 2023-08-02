
import os
import random
from pprint import pprint

import betfairutil
import dotenv
import matplotlib.pyplot as plt
import tennisim

from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
#from environments.tennis_markov import tennisMarkovSimulator
from environments.tennis_simulator import tennisSimulator
from src import data_processing
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




    ### AS SIMULATION USING TENNISIM
    a_s = 0.7
    b_s = 0.7
    #price_simulator = tennisMarkovSimulator.TennisMarkovSimulator(s=s, t=t)
    price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)

    simulator_framework = AvellanedaStoikovFramework()

    #strategy = RandomStrategy(range_offset=(0, 1))
    #strategy = FixedOffsetStrategy(offset=0.2)
    strategy = AvellanedaStoikovStrategy()

    simulator_framework.run_simulation(price_simulator=price_simulator, strategy=strategy, num_simulations=100)


if __name__=="__main__":
    main()