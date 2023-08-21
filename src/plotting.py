import os
import pickle

import dataframe_image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results_of_all_strategies_test(results_path, strategies_names_list, metrics_list):
    # Define the custom xlim and ylim for each metric.
    xlim_values = {
        'final_pnl': (-16, 16),
        "volatility": (0, 2),
        "mean_return": (-0.15, 0.15),
        "min_pnl": (-13, 0),
        "max_pnl": (0, 20),
        "sharpe_ratio": (-0.5, 0.8),
        "sortino_ratio": (-0.75, 2.5),
        "mean_inv_stake": (-18, 13)
    }

    ylim_values = {
        'final_pnl': (0, 2000),
        "volatility": (0, 2500),
        "mean_return": (0, 1500),
        "min_pnl": (0, 2500),
        "max_pnl": (0, 3000),
        "sharpe_ratio": (0, 2000),
        "sortino_ratio": (0, 2500),
        "mean_inv_stake": (0, 2300)
    }

    #f, ax = plt.subplots(figsize=(7, 3))
    #f.tight_layout()


    for metric in metrics_list:
        print(metric)
        plt.figure(figsize=(7, 3))
        for strategy in strategies_names_list:
            path = os.path.join(results_path, strategy, "result.pkl")
            with open(path, 'rb') as f:
                dict_result = pickle.load(f)
                data = [r for r in dict_result[metric]]
                label = strategy.replace("_0", " 0.") + " offset"
                print(len(data))
                sns.histplot(data, label=label)

        # Set plot labels and title based on current metric, and fetch the respective xlim and ylim from the dictionaries

        title = metric.replace("_", " ")
        plt.xlim(*xlim_values[metric])
        plt.ylim(*ylim_values[metric])
        plt.xlabel(f'{title}', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        #plt.title(f"{title} of baseline models")
        metric_path = os.path.join(results_path, f"{metric}_models")
        print(metric_path)
        plt.savefig(metric_path,
                    bbox_inches='tight'
                    )
        # plt.show()
        plt.close()





def plot_results_of_single_strategy_test(plot_path, dict_results):
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