import betfairutil
import numpy as np


def calculate_orders_rates_for_lob_simulation(num_mos,
                                              num_los,
                                              num_cos,
                                              num_updates,
                                              list_volume_mos,
                                              list_volume_los):

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



