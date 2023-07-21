
import os
from pprint import pprint

import dotenv

from src import data_processing


def main():
    dotenv.load_dotenv()
    data_directory = os.environ.get("DATA_DIRECTORY")


    ## EXTRACT ORDER INTENSITIES AND VOLUME RATES FOR LOB SIMULATION
    price_file_path = os.path.join(data_directory, "djokovic_match_odds/29/32060431/1.209205488.bz2")
    id_runner = 2249229
    dict_results = data_processing.extract_orders_and_volumes_from_price_file(price_file_path=price_file_path,
                                                     id_runner=id_runner)

    # num_mos_back = dict_results['back']["num_mos"]
    # num_los_back = dict_results['back']["num_los"]
    # num_cos_back = dict_results['back']["num_cos"]
    # num_updates_back = dict_results['back']["num_updates"]
    # list_volume_mos_back = dict_results['back']["list_volume_mos"]
    # list_volume_los_back = dict_results['back']["list_volume_los"]

    # num_mos_lay = dict_results['lay']["num_mos"]
    # num_los_lay = dict_results['lay']["num_los"]
    # num_cos_lay = dict_results['lay']["num_cos"]
    # num_updates_lay = dict_results['lay']["num_updates"]
    # list_volume_mos_lay = dict_results['lay']["list_volume_mos"]
    # list_volume_los_lay = dict_results['lay']["list_volume_los"]


    dict_rates_back = data_processing.calculate_orders_rates_for_lob_simulation(dict_extraction_results=dict_results['back'])
    dict_rates_lay = data_processing.calculate_orders_rates_for_lob_simulation(dict_extraction_results=dict_results['lay'])

    for name_rate, rate in dict_rates_back.items():
        print(f"{name_rate}: {rate}\n")

    for name_rate, rate in dict_rates_lay.items():
        print(f"{name_rate}: {rate}\n")





    #pprint(num_cos)






if __name__=="__main__":
    main()