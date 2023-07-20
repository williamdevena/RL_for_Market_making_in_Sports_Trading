import betfairutil
import numpy as np


def calculate_orders_rates_for_lob_simulation(num_mos,
                                              num_los,
                                              num_cos,
                                              num_updates,
                                              list_volume_mos,
                                              list_volume_los):
    """
    Calculates the order intensity rates (lambda) and order volume rates (alpha) for market orders, limit orders,
    and cancellation orders. These parameters are used in the simulation of a Limit Order Book (LOB) model.

    The function uses logarithmic calculations for the volume rates and ratio calculations for the intensity rates.

    The Markov Chain model that simulates the LOB, these parameters and their formulas follow
    the following work: Hult, H., & Kiessling, J. (2010). Algorithmic trading with Markov chains..

    Args:
        num_mos (int): The total count of market orders.
        num_los (dict): A dictionary containing the count of limit orders at different levels of the LOB.
                        Keys are level indices, and values are counts of orders at that level.
        num_cos (dict): A dictionary containing the count of cancellation orders at different levels of the LOB.
                        Keys are level indices, and values are counts of orders at that level.
        num_updates (int): The total number of updates in the LOB.
        list_volume_mos (list): A list of volumes for each market order.
        list_volume_los (list): A list of volumes for each limit order.

    Returns:
        dict: A dictionary containing the calculated parameters 'alpha_mos', 'alpha_los', 'lambda_mos', 'lamda_los',
        and 'lamda_cos'. 'alpha_mos' and 'alpha_los' are the volume rates for market and limit orders, respectively.
        'lambda_mos' is the intensity rate for market orders. 'lamda_los' and 'lamda_cos' are lists of intensity rates
        for limit and cancellation orders at each level of the LOB, respectively.
    """
    alpha_mos = np.log(np.mean(list_volume_mos)/(np.mean(list_volume_mos)-1))
    alpha_los = np.log(np.mean(list_volume_los)/(np.mean(list_volume_los)-1))

    lambda_mos = num_mos/num_updates
    lamda_los = [num_orders/num_updates for level, num_orders in num_los.items()]
    lamda_cos = [num_orders/num_updates for level, num_orders in num_cos.items()]

    return {"alpha_mos": alpha_mos,
            "alpha_los": alpha_los,
            "lambda_mos": lambda_mos,
            "lamda_los": lamda_los,
            "lamda_cos": lamda_cos}


def extract_orders_and_volumes_from_price_file(price_file_path, id_runner):
    """
    Process a Betfair price file to extract and count different types of orders
    (Market orders, Limit orders, and Cancellation orders) and their volumes for
    a specific runner in the market from both the 'lay' and 'back' sides of the
    Limit Order Book (LOB).

    The function is used in conjunction with 'calculate_orders_rates_for_lob_simulation'
    function to calibrate the parameters lambda (order arrival rate) and alpha (order size)
    for a Markov Chain model that simulates the LOB.

    Args:
        price_file_path (str): The file path of the Betfair price file to be processed.
        id_runner (int): The unique identifier of the runner for which the orders and volumes will be extracted.

    Returns:
        dict: A dictionary containing counts of market orders, limit orders, and cancellation orders,
        along with their volumes, for both 'lay' and 'back' sides. The keys of this dictionary are 'lay'
        and 'back', and the values are another dictionary with keys 'num_mos' (count of market orders),
        'num_los' (count of limit orders), 'num_cos' (count of cancellation orders), 'num_updates'
        (count of updates in the LOB), 'list_volume_mos' (list of volumes of market orders),
        and 'list_volume_los' (list of volumes of limit orders).
    """
    market_books = betfairutil.read_prices_file(price_file_path)
    # dict_trd_prices = {}
    # dict_last_trd = {}
    dict_result = {}

    for side_tuple in [('lay','availableToBack', 'atb'), ('back','availableToLay', 'atl')]:
        dict_available_volumes_lay = {}
        dict_traded_volume = {}

        num_mos = {}
        num_los = {}
        num_cos = {}
        num_updates = 0

        list_volume_mos = []
        list_volume_los = []

        for idx, mb in enumerate(market_books[:40000]):
            # if idx<1000:
            #         continue
            runner = [runner for runner in mb['runners'] if runner['selectionId']==id_runner][0]
            lob = runner['ex']
            lob_lay = lob[side_tuple[1]]
            traded_volume = lob['tradedVolume']
            update_list = mb['streaming_update']['rc']
            num_updates += 1

            # runner_updates = [update for update in update_list
            #                   if update['id']==id_runner]
            # trd_updates = [update for update in runner_updates if "trd" in update]
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
                                      "num_updates": num_updates,
                                      "list_volume_mos": list_volume_mos,
                                      "list_volume_los": list_volume_los
                                     }

    return dict_result



