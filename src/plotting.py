"""
This module contains all the necessary functions to plot data generated in other modules.
"""
import copy
import os
import pickle

import dataframe_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict


def plot_correlation_matrix_state_actions(corr_matrix: np.ndarray,
                                          plot_path: str,
                                          list_feature_names: List[str]) -> None:
    """
    Plots a correlation matrix for state and action features.

    Args:
        corr_matrix (np.ndarray): Correlation matrix to plot.
        plot_path (str): Path where the plot will be saved.
        list_feature_names (List[str]): List of feature names for the matrix.
    """
    corr_matrix = corr_matrix[5:]
    mask = copy.deepcopy(corr_matrix)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if j<(i+5):
                mask[i][j] = 0

    f, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(corr_matrix, square=False, annot=True,
                mask=mask,
                xticklabels=[feature for feature, _ in list_feature_names
                             if feature!='lay_offset'],
                yticklabels=['spread', 'back_offset', 'lay_offset'],
                vmin=-1,
                vmax=1)
    plt.savefig(plot_path)
    plt.close()



def plot_results_of_all_strategies_test(results_path: str,
                                        strategies_names_list: List[str],
                                        metrics_list: List[str]) -> None:
    """
    Plots the results of all strategies for the test set.

    Args:
        results_path (str): Path where the results are stored.
        strategies_names_list (List[str]): List of strategy names.
        metrics_list (List[str]): List of metrics to plot.
    """
    # Define the custom xlim and ylim for each metric.
    xlim_values = {
        'final_pnl': (-35, 35),
        "volatility": (0, 2),
        "mean_return": (-0.15, 0.15),
        "min_pnl": (-13, 0),
        "max_pnl": (0, 20),
        "sharpe_ratio": (-0.5, 0.8),
        "sortino_ratio": (-0.75, 2.5),
        "mean_inv_stake": (-18, 13)
    }
    ylim_values = {
        'final_pnl': (0, 3000),
        #'final_pnl': (0, 110),
        "volatility": (0, 2500),
        "mean_return": (0, 1500),
        "min_pnl": (0, 2500),
        "max_pnl": (0, 3000),
        "sharpe_ratio": (0, 2000),
        "sortino_ratio": (0, 2500),
        "mean_inv_stake": (0, 2300)
    }

    for metric in metrics_list:
        print(metric)
        plt.figure(figsize=(7, 3))
        for strategy in strategies_names_list:
            path = os.path.join(results_path, strategy, "result.pkl")
            with open(path, 'rb') as f:
                dict_result = pickle.load(f)
                data = [r for r in dict_result[metric]]
                label = strategy.split("_")[0]
                sns.histplot(data, label=label,
                             bins=300
                             )
        title = metric.replace("_", " ")
        plt.xlim(*xlim_values[metric])
        plt.ylim(*ylim_values[metric])
        plt.xlabel(f'{title}', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper right')
        metric_path = os.path.join(results_path, f"{metric}_models")
        print(metric_path)
        plt.savefig(metric_path,
                    bbox_inches='tight'
                    )
        plt.close()





def plot_results_of_single_strategy_test(plot_path: str,
                                         dict_results: Dict[str, np.ndarray]) -> None:
    """
    Plots the results of a single strategy for the test set.

    Args:
        plot_path (str): Path where the plots will be saved.
        dict_results (Dict[str, np.ndarray]): Dictionary containing result arrays for various metrics.
    """
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    final_pnl = dict_results['final_pnl']
    mean_return = dict_results['mean_return']
    volatility_returns = dict_results['volatility']
    min_pnl = dict_results['min_pnl']
    max_pnl = dict_results['max_pnl']
    sharpe_ratio = dict_results['sharpe_ratio']
    sortino_ratio = dict_results['sortino_ratio']
    mean_inv_stake = dict_results['mean_inv_stake']

    plt.hist(final_pnl, bins=50)
    plt.xlabel('Final PnL')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "final_pnl"))
    plt.close()

    plt.hist(mean_return, bins=50)
    plt.xlabel('Mean Return')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "mean_return"))
    plt.close()

    plt.hist(volatility_returns, bins=50)
    plt.xlabel('Volatility')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "volatility"))
    plt.close()

    plt.hist(min_pnl, bins=50)
    plt.xlabel('Min PnL (Max Loss)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "min_pnl"))
    plt.close()

    plt.hist(max_pnl, bins=50)
    plt.xlabel('Max PnL')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "max_pnl"))
    plt.close()

    plt.hist(sharpe_ratio, bins=50)
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "sharpe_ratio"))
    plt.close()

    plt.hist(sortino_ratio, bins=50)
    plt.xlabel('Sortino Ratio')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "sortino_ratio"))
    plt.close()

    plt.hist(mean_inv_stake, bins=50)
    plt.xlabel('Mean Inventory Stake')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plot_path, "mean_inv_stake"))
    plt.close()

    df_final_pnl = pd.DataFrame(final_pnl, columns=['Final PnL'])
    df_mean_return = pd.DataFrame(mean_return, columns=['Mean Return'])
    df_volatility = pd.DataFrame(volatility_returns, columns=['Volatility'])
    df_min_pnl = pd.DataFrame(min_pnl, columns=['Min PnL'])
    df_max_pnl = pd.DataFrame(max_pnl, columns=['Max PnL'])
    df_sharpe = pd.DataFrame(sharpe_ratio, columns=['Sharpe ratio'])
    df_sortino = pd.DataFrame(sortino_ratio, columns=['Sortino ratio'])
    df_mean_inv_stake = pd.DataFrame(mean_inv_stake, columns=['Mean Inv. Stake'])

    final_df_describe = pd.concat(objs=[df_final_pnl.describe(),
                                        df_mean_return.describe(),
                                        df_volatility.describe(),
                                        df_min_pnl.describe(),
                                        df_max_pnl.describe(),
                                        df_sharpe.describe(),
                                        df_sortino.describe(),
                                        df_mean_inv_stake.describe(),],
                                    axis='columns')

    dataframe_image.export(final_df_describe, os.path.join(plot_path, 'all_describes.png'))
    with open(os.path.join(plot_path, "latex_table"), 'w') as f:
        print(final_df_describe.to_latex(), file=f)

    with open(os.path.join(plot_path,'result.pkl'), 'wb') as f:
            pickle.dump({'final_pnl': final_pnl,
                         'mean_return': mean_return,
                         'volatility': volatility_returns,
                         'min_pnl': min_pnl,
                         'max_pnl': max_pnl,
                         'sharpe_ratio': sharpe_ratio,
                         'sortino_ratio': sortino_ratio,
                         'mean_inv_stake': mean_inv_stake,},
                        f)