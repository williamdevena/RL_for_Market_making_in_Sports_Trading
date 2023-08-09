import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_it

from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
from environments.tennis_simulator import tennisSimulator
from src import plotting
from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy
from strategy.fixedOffsetStrategy import FixedOffsetStrategy
from strategy.randomStrategy import RandomStrategy


def test_fixed_offset_strategy(plot_path):

    a_s = 0.65
    b_s = 0.65

    tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    #tennis_probs = [0.60, 0.62]
    k_range = range(3, 13)
    #k_range = range(3, 5)
    offset_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))
    print(len(possible_combinations))

    num_simulations = 100

    final_pnl = []
    volatility_pnl = []
    min_pnl = []
    max_pnl = []

    final_pickle_saving = []
    vol_pickle_saving = []
    min_pickle_saving = []
    max_pickle_saving = []

    x = 0

    # for a_s, b_s in list(itertools.product(tennis_probs, repeat=2)):
    #     for k in k_range:
    for offset in offset_range:
        for_saving_final_pnl = []
        for_saving_vol = []
        for_saving_min = []
        for_saving_max = []
        for a_s, b_s, k in alive_it(possible_combinations):
            fixed_strategy = FixedOffsetStrategy(offset=offset)

            price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)
            simulator_framework = AvellanedaStoikovFramework(k=k)


            sim_results = simulator_framework.run_single_simulation(price_simulator=price_simulator,
                                                                    strategy=fixed_strategy,
                                                                    num_simulations=num_simulations,
                                                                    plotting=False)
            final_pnl.extend(sim_results['final_pnl'])
            volatility_pnl.extend(sim_results['volatility'])
            min_pnl.extend(sim_results['min_pnl'])
            max_pnl.extend(sim_results['max_pnl'])

            for_saving_final_pnl.extend(sim_results['final_pnl'])
            for_saving_vol.extend(sim_results['volatility'])
            for_saving_final_pnl.extend(sim_results['min_pnl'])
            for_saving_final_pnl.extend(sim_results['max_pnl'])

        final_pickle_saving.append((offset, for_saving_final_pnl))
        vol_pickle_saving.append((offset, for_saving_vol))
        min_pickle_saving.append((offset, for_saving_min))
        max_pickle_saving.append((offset, for_saving_max))


    with open(os.path.join(plot_path, 'results_for_offset.pkl'), 'wb') as f:
        pickle.dump({'final_pnl': final_pickle_saving,
                        'volatility': vol_pickle_saving,
                        'min_pnl': min_pickle_saving,
                        'max_pnl': max_pickle_saving},
                    f)


    plotting.plot_results_of_strategy_tests(final_pnl=final_pnl,
                                            volatility_pnl=volatility_pnl,
                                            min_pnl=min_pnl,
                                            max_pnl=max_pnl,
                                            plot_path=plot_path)





def test_strategies(plot_path, strategy):

    a_s = 0.65
    b_s = 0.65

    tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    #tennis_probs = [0.60, 0.62]
    k_range = range(3, 13)
    #k_range = range(3, 5)
    offset_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    num_simulations = 100

    final_pnl = []
    volatility_pnl = []
    min_pnl = []
    max_pnl = []

    for a_s, b_s, k in alive_it(possible_combinations):

        price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)
        simulator_framework = AvellanedaStoikovFramework(k=k)


        sim_results = simulator_framework.run_single_simulation(price_simulator=price_simulator,
                                                                strategy=strategy,
                                                                num_simulations=num_simulations,
                                                                plotting=False)
        final_pnl.extend(sim_results['final_pnl'])
        volatility_pnl.extend(sim_results['volatility'])
        min_pnl.extend(sim_results['min_pnl'])
        max_pnl.extend(sim_results['max_pnl'])


    plotting.plot_results_of_strategy_tests(final_pnl=final_pnl,
                                            volatility_pnl=volatility_pnl,
                                            min_pnl=min_pnl,
                                            max_pnl=max_pnl,
                                            plot_path=plot_path)