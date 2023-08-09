
import os
import pickle
import random
import time
from pprint import pprint

import betfairutil
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import tennisim

from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
#from environments.tennis_markov import tennisMarkovSimulator
from environments.tennis_simulator import tennisSimulator
from src import data_processing, testing
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

    # price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)

    # simulator_framework = AvellanedaStoikovFramework(k=7)

    # strategy = RandomStrategy(range_offset=(0, 1))
    # #strategy = FixedOffsetStrategy(offset=0.1)
    # #strategy = AvellanedaStoikovStrategy()

    # num_simulations = 100
    # start_time = time.time()

    # plot_path = "./plots/random"

    # simulator_framework.run_simulation(price_simulator=price_simulator,
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
    plot_path = "./plots/fixed"

    #random_strategy = RandomStrategy(range_offset=(0, 1))
    fixed_strategy = FixedOffsetStrategy(offset=0.2)
    #as_strategy = AvellanedaStoikovStrategy()

    testing.test_strategies(plot_path=plot_path,
                            strategy=fixed_strategy)

    # plot_path = "./plots/fixed"
    # testing.test_fixed_offset_strategy(plot_path=plot_path)






    # ### LOADING PICKLE FILES

    # with open('./plots/random/result.pkl', 'rb') as f:
    #     dict_result = pickle.load(f)

    # final_pnl = [pnl for pnl in dict_result['final_pnl']
    #              if pnl<500 and pnl>-500]
    # volat = [vol for vol in dict_result['volatility']
    #          if vol<200]
    # min_pnl = [min for min in dict_result['min_pnl']
    #            if min>-500]
    # max_pnl = [max for max in dict_result['max_pnl']
    #            if max<500]

    # plt.hist(final_pnl, bins=50)
    # plt.xlabel('Final PnL')
    # plt.ylabel('Frequency')
    # plt.title("Random offset model")
    # plt.savefig("./plots/random/final_pnl_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(volat, bins=50)
    # plt.xlabel('Volatility')
    # plt.ylabel('Frequency')
    # plt.title("Random offset model")
    # plt.savefig("./plots/random/vol_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(min_pnl, bins=50)
    # plt.xlabel('Min PnL')
    # plt.ylabel('Frequency')
    # plt.title("Random offset model")
    # plt.savefig("./plots/random/min_pnl_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(max_pnl, bins=50)
    # plt.xlabel('Max PnL')
    # plt.ylabel('Frequency')
    # plt.title("Random offset model")
    # plt.savefig("./plots/random/max_pnl_adjusted")
    # #plt.show()
    # plt.close()







    # with open('./plots/as/result.pkl', 'rb') as f:
    #     dict_result = pickle.load(f)

    # final_pnl = [pnl for pnl in dict_result['final_pnl']
    #              if pnl<500 and pnl>-500]
    # volat = [vol for vol in dict_result['volatility']
    #          if vol<200]
    # min_pnl = [min for min in dict_result['min_pnl']
    #            if min>-500]
    # max_pnl = [max for max in dict_result['max_pnl']
    #            if max<500]

    # plt.hist(final_pnl, bins=50)
    # plt.xlabel('Final PnL')
    # plt.ylabel('Frequency')
    # plt.title("Avellaneda-Stoikov model")
    # plt.savefig("./plots/as/final_pnl_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(volat, bins=50)
    # plt.xlabel('Volatility')
    # plt.ylabel('Frequency')
    # plt.title("Avellaneda-Stoikov model")
    # plt.savefig("./plots/as/vol_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(min_pnl, bins=50)
    # plt.xlabel('Min PnL')
    # plt.ylabel('Frequency')
    # plt.title("Avellaneda-Stoikov model")
    # plt.savefig("./plots/as/min_pnl_adjusted")
    # #plt.show()
    # plt.close()

    # plt.hist(max_pnl, bins=50)
    # plt.xlabel('Max PnL')
    # plt.ylabel('Frequency')
    # plt.title("Avellaneda-Stoikov model")
    # plt.savefig("./plots/as/max_pnl_adjusted")
    # #plt.show()
    # plt.close()






if __name__=="__main__":
    main()