import os
import pickle

import dataframe_image
import matplotlib.pyplot as plt
import pandas as pd


def plot_results_of_strategy_tests(plot_path, dict_results):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    final_pnl = dict_results['final_pnl']
    volatility_pnl = dict_results['volatility_pnl']
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

    plt.hist(volatility_pnl, bins=50)
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

    df_final_pnl = pd.DataFrame(final_pnl, columns=['Final PnL'])
    df_volatility = pd.DataFrame(volatility_pnl, columns=['Volatility'])
    df_min_pnl = pd.DataFrame(min_pnl, columns=['Min PnL'])
    df_max_pnl = pd.DataFrame(max_pnl, columns=['Max PnL'])
    df_sharpe = pd.DataFrame(sharpe_ratio, columns=['Sharpe ratio'])
    df_sortino = pd.DataFrame(sortino_ratio, columns=['Sortino ratio'])

    final_df_describe = pd.concat(objs=[df_final_pnl.describe(),
                                        df_volatility.describe(),
                                        df_min_pnl.describe(),
                                        df_max_pnl.describe(),
                                        df_sharpe.describe(),
                                        df_sortino.describe(),],
                                    axis='columns')

    dataframe_image.export(final_df_describe, os.path.join(plot_path, 'all_describes.png'))
    print(final_df_describe.to_latex())

    with open(os.path.join(plot_path,'result.pkl'), 'wb') as f:
            pickle.dump({'final_pnl': final_pnl,
                         'volatility': volatility_pnl,
                         'min_pnl': min_pnl,
                         'max_pnl': max_pnl,
                         'sharpe_ratio': sharpe_ratio,
                         'sortino_ratio': sortino_ratio,},
                        f)