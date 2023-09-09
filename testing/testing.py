"""
This moudle contains the functions to execute the testing of the RL agents.
"""
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import alive_it

sys.path.append("..")
from environments.gym_env import sportsTradingEnvironment
from src import plotting


def test_correlations_state_vars_and_actions(model, mode, num_simul_per_combin, plot_path):
    """
    Test correlations (pearson and kendall correlations) between state variables and actions
    using a given RL model. Used to analyse and undestrand the "logic" of the agents.

    Args:
        model: Trained RL model to be tested.
        mode (str): Specifies the testing mode. Either "long" for comprehensive testing or
                    any other value for shorter testing.
        num_simul_per_combin (int): Number of simulations to run for each combination of
                                    environment parameters.
        plot_path (str): Path to save the plotted correlation matrices.

    Returns:
        None. However, correlation matrices plots are saved to the specified directory.
    """
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))
    list_pearson_corr_matrices = []
    list_kendall_corr_matrices = []

    for a_s, b_s, k in alive_it(possible_combinations):
        for x in range(num_simul_per_combin):
            env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                    b_s=b_s,
                                                                    k=k,
                                                                    mode='fixed')

            dict_results = test_rl_agent_single_episode(model=model,
                                        env=env,
                                        debug=False,
                                        plot_results=False)
            dict_results['price'] = dict_results['price'][:-1]
            df_results = pd.DataFrame.from_dict(dict_results)
            pearson_corr_matrix = df_results.corr(method='pearson')
            kendall_corr_matrix = df_results.corr(method='kendall')
            list_pearson_corr_matrices.append(pearson_corr_matrix)
            list_kendall_corr_matrices.append(kendall_corr_matrix)

    pearson_mean_matrix = np.nanmean(list_pearson_corr_matrices, axis=0)
    kendall_mean_matrix = np.nanmean(list_kendall_corr_matrices, axis=0)

    plotting.plot_correlation_matrix_state_actions(corr_matrix=pearson_mean_matrix,
                                                   plot_path=plot_path+"_pearson",
                                                   list_feature_names=dict_results.items())

    plotting.plot_correlation_matrix_state_actions(corr_matrix=kendall_mean_matrix,
                                                   plot_path=plot_path+"_kendall",
                                                   list_feature_names=dict_results.items())


def test_rl_agent_all_combinations(model, num_simulations_per_combination, plot_path, mode="long", debug=False):
    """
    Test an RL agent over all combinations of environment parameters ("All_comb" testing
    procedure).

    Args:
        model: Trained RL model to be tested.
        num_simulations_per_combination (int): Number of simulations for each combination.
        plot_path (str): Path to save the plotted results.
        mode (str, optional): Specifies the testing mode. Defaults to "long".
        debug (bool, optional): If True, will print debug information. Defaults to False.

    Returns:
        None. Plots results of the test.
    """
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []
    for a_s, b_s, k in alive_it(possible_combinations):
        env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                b_s=b_s,
                                                                k=k,
                                                                mode='fixed')
        sim_results = test_rl_agent_multiple_episods(num_episodes=num_simulations_per_combination,
                                       model=model,
                                       env=env,
                                       plot_results=False)

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


def test_rl_agent_single_episode(model, env, debug=False, plot_results=True):
    """
    Test an RL agent on a single episode.

    Args:
        model: Trained RL model to be tested.
        env: Custom Gym environment for sports trading.
        debug (bool, optional): If True, will print debug information. Defaults to False.
        plot_results (bool, optional): If True, results will be plotted. Defaults to True.

    Returns:
        dict: Dictionary containing various metrics and series from the test.
    """
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if debug:
            print(f"Obs: {obs}\nAction: {action}")

    spread = [a+b for (a,b) in list(zip(env.back_offsets, env.lay_offsets))]

    if plot_results:
        f = plt.figure(figsize=(10, 10))
        f.add_subplot(2, 2, 1)
        plt.plot(env.price, label="mid-price")
        plt.plot(env.back_prices, label="back price")
        plt.plot(env.lay_prices, label="lay price")
        plt.legend()

        f.add_subplot(2, 2, 2)
        plt.plot(env.back_offsets, label="back offset")
        plt.plot(env.lay_offsets, label="lay offset")
        plt.plot(spread, label="spread")
        plt.legend()

        f.add_subplot(2, 2, 3)
        plt.plot(env.list_pnl, label="PnL")
        plt.plot(env.list_inventory_stake, label="Stake")
        plt.plot(env.list_inventory_odds, label="Odds")
        plt.legend()

        f.add_subplot(2, 2, 4)
        plt.plot(env.list_volatility_indicator, label="Volatility indicator")
        plt.plot(env.list_momentum_indicator, label="Momentum indicator")
        plt.legend()
        plt.show()

    return {'stake': env.list_inventory_stake,
            'odds': env.list_inventory_odds,
            'price': env.price,
            'momentum': env.list_momentum_indicator,
            'volatility': env.list_volatility_indicator,
            'spread': spread,
            'back_offset': env.back_offsets,
            'lay_offset': env.lay_offsets}




def test_rl_agent_multiple_episods(num_episodes, model, env, plot_results=True, plot_path=None, debug=False):
    """
    Test an RL agent over multiple episodes.

    Args:
        num_episodes (int): Number of episodes for the test.
        model: Trained RL model or a predefined strategy ('random', 'fixed_02', 'fixed_05', 'fixed_08') to be tested.
        env: Custom Gym environment for sports trading.
        plot_results (bool, optional): If True, results will be plotted. Defaults to True.
        plot_path (str, optional): Path to save the plotted results. Defaults to None.
        debug (bool, optional): If True, will print debug information. Defaults to False.

    Returns:
        dict: Dictionary containing various aggregated metrics from the tests.
    """
    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            if model=="random":
                action = env.action_space.sample()
            elif model=='fixed_02':
                action = 22
            elif model=='fixed_05':
                action = 55
            elif model=='fixed_08':
                action = 88
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if debug:
                print(f"Obs: {obs}\nAction: {action}")


        final_pnl.append(env.list_pnl[-1])
        min_pnl.append(np.min(env.list_pnl))
        max_pnl.append(np.max(env.list_pnl))

        pnl_series = pd.Series(env.list_pnl)
        returns = pnl_series.pct_change()
        returns = returns.replace(to_replace=[np.inf, np.NINF], value=np.nan)
        returns = returns.interpolate()
        returns = returns.replace(np.nan, 0)

        std_returns = np.std(returns)
        downside_std = np.nanstd(np.clip(returns, np.NINF, 0, out=None))
        mean_ret = np.mean(returns)
        mean_return.append(mean_ret)
        volatility_returns.append(std_returns)

        if mean_ret==0 or std_returns==0:
            sharpe_ratio.append(0)
        else:
            sharpe_ratio.append(mean_ret/std_returns)

        if mean_ret==0 or downside_std==0:
            sortino_ratio.append(0)
        else:
            sortino_ratio.append(mean_ret/downside_std)

        mean_inv_stake.append(np.mean(env.list_inventory_stake))

    if plot_results:
        plotting.plot_results_of_single_strategy_test(plot_path=plot_path,
                                                      dict_results={'final_pnl': final_pnl,
                                                                    'mean_return': mean_return,
                                                                    'volatility': volatility_returns,
                                                                    'min_pnl': min_pnl,
                                                                    'max_pnl': max_pnl,
                                                                    'sharpe_ratio': sharpe_ratio,
                                                                    'sortino_ratio': sortino_ratio,
                                                                    'mean_inv_stake': mean_inv_stake})

    return {'final_pnl': final_pnl,
            'mean_return': mean_return,
            'volatility': volatility_returns,
            'min_pnl': min_pnl,
            'max_pnl': max_pnl,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'mean_inv_stake': mean_inv_stake,}

