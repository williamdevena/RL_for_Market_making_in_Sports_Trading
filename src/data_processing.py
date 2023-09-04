import betfairutil
import numpy as np


def extract_orders_rates_for_lob_simulation(market_books, id_runner):
    """
    Extracts the orders and volumes from a list of market books, calculates order intensity rates (lambda) and order volume rates
    (alpha) for the simulation of a Limit Order Book (LOB) model, for both 'back' and 'lay' sides of the market.

    This function is a high-level wrapper that calls 'extract_orders_and_volumes_from_price_file' and
    'calculate_orders_rates_for_lob_simulation' functions, to extract necessary data from the market books and then
    compute the order intensity rates and order volume rates.

    Args:
        market_books (list): A list of market book dictionaries containing the price file data.
        id_runner (int): Identifier for the runner (a participant in the event, e.g., a horse in a horse race).

    Returns:
        tuple: A tuple containing two dictionaries, one for 'back' and one for 'lay'. Each dictionary contains the
        calculated parameters 'alpha_mos', 'alpha_los', 'lambda_mos', 'lamda_los', and 'lamda_cos'. 'alpha_mos' and
        'alpha_los' are the volume rates for market and limit orders, respectively. 'lambda_mos' is the intensity rate
        for market orders. 'lamda_los' and 'lamda_cos' are lists of intensity rates for limit and cancellation orders
        at each level of the LOB, respectively.
    """

    dict_results = extract_orders_and_volumes_from_price_file(market_books=market_books,
                                                            id_runner=id_runner)

    publish_time_first_mb = market_books[0]['publishTime']
    publish_time_last_mb = market_books[-1]['publishTime']
    total_seconds = (publish_time_last_mb - publish_time_first_mb)/1000

    print(total_seconds)

    dict_rates_back = calculate_orders_rates_for_lob_simulation(dict_extraction_results=dict_results['back'], total_seconds=total_seconds)
    dict_rates_lay = calculate_orders_rates_for_lob_simulation(dict_extraction_results=dict_results['lay'], total_seconds=total_seconds)

    return dict_rates_back, dict_rates_lay




def calculate_orders_rates_for_lob_simulation(dict_extraction_results, total_seconds):
    """
    Calculates the order intensity rates (lambda) and order volume rates (alpha) for market orders, limit orders,
    and cancellation orders. These parameters are used in the simulation of a Limit Order Book (LOB) model.

    The function uses logarithmic calculations for the volume rates and ratio calculations for the intensity rates.

     The Markov Chain model that simulates the LOB, these parameters and their formulas follow
    the following work: Hult, H., & Kiessling, J. (2010). Algorithmic trading with Markov chains.

    Args:
        dict_extraction_results (dict): A dictionary containing the extraction results from the price file.
            "num_mos" (int): The total count of market orders.
            "num_los" (dict): A dictionary containing the count of limit orders at different levels of the LOB.
                              Keys are level indices, and values are counts of orders at that level.
            "num_cos" (dict): A dictionary containing the count of cancellation orders at different levels of the LOB.
                              Keys are level indices, and values are counts of orders at that level.
            "list_volume_mos" (list): A list of volumes for each market order.
            "list_volume_los" (list): A list of volumes for each limit order.
        total_seconds (float): total time range in seconds

    Returns:
        dict: A dictionary containing the calculated parameters 'alpha_mos', 'alpha_los', 'lambda_mos', 'lamda_los',
        and 'lamda_cos'. 'alpha_mos' and 'alpha_los' are the volume rates for market and limit orders, respectively.
        'lambda_mos' is the intensity rate for market orders. 'lamda_los' and 'lamda_cos' are lists of intensity rates
        for limit and cancellation orders at each level of the LOB, respectively.
    """
    num_mos = dict_extraction_results["num_mos"]
    num_los = dict_extraction_results["num_los"]
    num_cos = dict_extraction_results["num_cos"]
#     num_updates = dict_extraction_results["num_updates"]
    list_volume_mos = dict_extraction_results["list_volume_mos"]
    list_volume_los = dict_extraction_results["list_volume_los"]

#     num_mos_lay = dict_results['lay']["num_mos"]
#     num_los_lay = dict_results['lay']["num_los"]
#     num_cos_lay = dict_results['lay']["num_cos"]
#     num_updates_lay = dict_results['lay']["num_updates"]
#     list_volume_mos_lay = dict_results['lay']["list_volume_mos"]
#     list_volume_los_lay = dict_results['lay']["list_volume_los"]
    alpha_mos = np.log(np.mean(list_volume_mos)/(np.mean(list_volume_mos)-1))
    alpha_los = np.log(np.mean(list_volume_los)/(np.mean(list_volume_los)-1))

    lambda_mos = num_mos/total_seconds
    lamda_los = [num_orders/total_seconds for level, num_orders in num_los.items()]
    lamda_cos = [num_orders/total_seconds for level, num_orders in num_cos.items()]

    return {"alpha_mos": alpha_mos,
            "alpha_los": alpha_los,
            "lambda_mos": lambda_mos,
            "lamda_los": lamda_los,
            "lamda_cos": lamda_cos}


def extract_orders_and_volumes_from_price_file(market_books, id_runner):
    """
    Parses a list of market books to count different types of orders
    (Market orders, Limit orders, and Cancellation orders) and their volumes for
    a specific runner in the market from both the 'lay' and 'back' sides of the
    Limit Order Book (LOB).

    The function is used in conjunction with 'calculate_orders_rates_for_lob_simulation'
    function to calibrate the parameters lambda (order arrival rate) and alpha (order size)
    for a Markov Chain model that simulates the LOB.

    Args:
        market_books (list): The file path of the Betfair price file to be processed.
        id_runner (int): The unique identifier of the runner for which the orders and volumes will be extracted.

    Returns:
        dict: A dictionary containing counts of market orders, limit orders, and cancellation orders,
        along with their volumes, for both 'lay' and 'back' sides. The keys of this dictionary are 'lay'
        and 'back', and the values are another dictionary with keys 'num_mos' (count of market orders),
        'num_los' (count of limit orders), 'num_cos' (count of cancellation orders),
        'list_volume_mos' (list of volumes of market orders), and 'list_volume_los' (list of volumes
        of limit orders).
    """
    dict_result = {}

    for side_tuple in [('lay','availableToBack', 'atb'), ('back','availableToLay', 'atl')]:
        dict_available_volumes_lay = {}
        dict_traded_volume = {}

        num_mos = {}
        num_los = {}
        num_cos = {}
        #num_updates = 0

        list_volume_mos = []
        list_volume_los = []

        for idx, mb in enumerate(market_books[:40000]):
            runner = [runner for runner in mb['runners'] if runner['selectionId']==id_runner][0]
            lob = runner['ex']
            lob_lay = lob[side_tuple[1]]
            traded_volume = lob['tradedVolume']
            update_list = mb['streaming_update']['rc']

            lay_updates = [update for update in update_list if update['id']==id_runner and side_tuple[2] in update]

            for lay_update in lay_updates:
                    for price_volume_tuple in lay_update[side_tuple[2]]:
                            price = price_volume_tuple[0]
                            volume = price_volume_tuple[1]

                            for idx, price_volume_dict in enumerate(lob_lay):
                                    if price_volume_dict['price']==price:
                                        level=idx

                            new_traded_volume_list = [price_volume_dict for price_volume_dict in traded_volume
                                                if price_volume_dict['price']==price]

                            if len(new_traded_volume_list)==0:
                                    new_traded_volume = 0
                            else:
                                    new_traded_volume = new_traded_volume_list[0]['size']

                            if dict_available_volumes_lay.get(price, 0) > volume:
                                    if dict_traded_volume.get(price, 0)==new_traded_volume:
                                        # CANCELLATION ORDER
                                        num_cos[level] = num_cos.get(level, 0) + 1
                                    elif dict_traded_volume.get(price, 0)<new_traded_volume:
                                        # MARKET ORDER
                                        num_mos[level] = num_mos.get(level, 0) + 1
                                        list_volume_mos.append(abs(dict_available_volumes_lay.get(price, 0)-volume))
                                    ## IF TOT. VOLUME TRADED DECREASES (SHOULDN'T HAPPEN)
                                    else:
                                        pass
                            else:
                                    # LIMIT ORDER
                                    num_los[level] = num_los.get(level, 0) + 1
                                    list_volume_los.append(abs(dict_available_volumes_lay.get(price, 0)-volume))

                            dict_available_volumes_lay[price] = volume

            for price_volume_dict in traded_volume:
                    dict_traded_volume[price_volume_dict['price']] = price_volume_dict['size']

        num_mos_tot = sum([num_orders for level, num_orders in num_mos.items()])
        dict_result[side_tuple[0]] = {"num_mos": num_mos_tot,
                                      "num_los": num_los,
                                      "num_cos": num_cos,
                                      "list_volume_mos": list_volume_mos,
                                      "list_volume_los": list_volume_los
                                     }

    return dict_result





