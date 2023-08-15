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


def test_strategies(plot_path, strategy, num_simulations_per_combination):
    tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    k_range = range(3, 13)

    # tennis_probs = [0.60, 0.62]
    # k_range = range(3, 5)

    #offset_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    #num_simulations = 5
    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []

    for a_s, b_s, k in alive_it(possible_combinations):
        price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)
        simulator_framework = AvellanedaStoikovFramework(k=k)

        sim_results = simulator_framework.run_single_simulation(price_simulator=price_simulator,
                                                                strategy=strategy,
                                                                num_simulations=num_simulations_per_combination,
                                                                plotting=False)
        final_pnl.extend(sim_results['final_pnl'])
        mean_return.extend(sim_results['mean_return'])
        volatility_returns.extend(sim_results['volatility'])
        min_pnl.extend(sim_results['min_pnl'])
        max_pnl.extend(sim_results['max_pnl'])
        sharpe_ratio.extend(sim_results['sharpe_ratio'])
        sortino_ratio.extend(sim_results['sortino_ratio'])
        mean_inv_stake.extend(sim_results['mean_inv_stake'])


    plotting.plot_results_of_single_strategy_test(plot_path=plot_path,
                                            dict_results={'final_pnl': final_pnl,
                                                          'mean_return': mean_return,
                                                        'volatility': volatility_returns,
                                                        'min_pnl': min_pnl,
                                                        'max_pnl': max_pnl,
                                                        'sharpe_ratio': sharpe_ratio,
                                                        'sortino_ratio': sortino_ratio,
                                                        'mean_inv_stake': mean_inv_stake}
                                            )