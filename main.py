
import os
from pprint import pprint

import betfairutil
import dotenv
import matplotlib.pyplot as plt

from environments.avellaneda_stoikov import avellaneda_stoikov
from environments.tennis_markov import tennisMarkovSimulator
from src import data_processing
from utils import pricefileutils


def main():
    dotenv.load_dotenv()
    data_directory = os.environ.get("DATA_DIRECTORY")


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

    # for x in range(100):
    #     prob_list, odds_list, games_idx = simulator.simulate()
    #     #plt.plot(prob_list)
    #     plt.plot(odds_list)
    #     simulator.restart()

    # #plt.ylim(0.9, 10)
    # plt.show()



    #### AVELLANEDA-STOIKOV WITH TENNIS SIMULATOR
    s = 0.5
    t = 0.5
    simulator = tennisMarkovSimulator.TennisMarkovSimulator(s=s, t=t)
    prob_list, odds_list, games_idx = simulator.simulate()

    avellaneda_stoikov.run(s=odds_list)




if __name__=="__main__":
    main()