import math
import os
import random

import dataframe_image as dfi
#import brownian as bm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_it

from baselines.avellanedaStoikovStrategy import AvellanedaStoikovStrategy

"""
High-frequency trading in a limit order book, Marco Avellaneda & Sasha Stoikov
paper url: https://www.researchgate.net/publication/24086205_High_Frequency_Trading_in_a_Limit_Order_Book

Some model limitations, discussed: https://quant.stackexchange.com/questions/36400/avellaneda-stoikov-market-making-model
Parameter fitting: https://quant.stackexchange.com/questions/36073/how-does-one-calibrate-lambda-in-a-avellaneda-stoikov-market-making-problem
"""


class AvellanedaStoikovFramework():
    """
    This class is used for running simulations of market-making strategies using the Avellaneda-Stoikov (AS)
    framework (Avellaneda M, Stoikov S. High-frequency trading in a limit order book. Quantitative Finance. 2008).
    It provides a method for running simulations over a price time series and plotting the results.

    Note: here we use a variation of the AS framework where the price is not modelled as a brownian motion, but
    the price simulator, which could be a brownian motion or a Markov model 8in case of tennis, is passed as a
    parameter. Hence we can see this class as a generalization of the AS framework.

    Attributes:
        k (float): A parameter of the Avellaneda-Stoikov framework that refers to the arrival of orders and the
                   prob. of getting the limit orders filled (the bigger k the least probable it is to get the
                   orders filled). The default value is 20.0.
        T (float): The total time until the market closes in the Avellaneda-Stoikov model.
                   The default value is 1.0.
                   Note: The total time T is divided in timesteps of dT=T/N where N=len(price time series)

    Methods:
        run_simulation(price_simulator, strategy, num_simulations=100): Executes the simulations and
                                                                         plots the results.
    """

    def __init__(self, k=20.0, T=1.0):
        self.k = k
        self.T = T


    def calculate_cash_out(self, stake, odds, current_odds):
        current_stake = ((odds+1)*stake)/(current_odds+1)
        cash_out = (stake*odds) - (current_stake*current_odds)

        return cash_out


    def combine_bets(self, list_bets):
        stake = sum([bet['stake'] for bet in list_bets])

        if stake==0:
            print("Total stake can't be 0")
            return False

        odds = sum([(bet['odds'] * (bet['stake']/stake)) for bet in list_bets])

        return {'stake': stake,
                'odds': odds}


    def run_single_simulation(self,
                       price_simulator,
                       strategy,
                       num_simulations=100,
                       plotting=False,
                       plot_path=None):
        """
        Run simulations of a market making strategy in a sports trading environment. The simulation results
        are plotted and some statistical information about the performance of the strategy is printed.

        Args:
            price_simulator (array_like): A TennisMarkovSimulator object, used to simulate the mid-price
                time series.
            strategy (Object): An instance of a Strategy class which defines the quotes method.
            num_simulations (int, optional): The number of simulations to run. Defaults to 100.
            plotting (bool): If True at the end of the simulation, the fucntion plot useful graphs of the
                simulation.
            plot_path (str): Used if 'plotting'=True.

        Returns: None. This function doesn't return anything but  if 'plotting'==True generates graphs
            of the simulation results.
        """
        if plot_path and not os.path.exists(plot_path):
            os.makedirs(plot_path)

        final_pnl = np.zeros((num_simulations))
        volatility_returns = np.zeros((num_simulations))
        min_pnl = np.zeros((num_simulations))
        max_pnl = np.zeros((num_simulations))
        mean_return = np.zeros((num_simulations))
        sharpe_ratio = np.zeros((num_simulations))
        sortino_ratio = np.zeros((num_simulations))
        mean_inv_stake = np.zeros((num_simulations))

        list_prices = []

        for i_sim in range(num_simulations):
            _, price = price_simulator.simulate()
            list_prices.append(price)

            N = len(price)
            #dt = self.T/N
            dt = 0.01
            t = np.linspace(0.0, N*dt, N)
            # Wealth
            pnl = np.zeros((N+1))
            pnl[0] = 0
            # Cash
            x = np.zeros((N+2))
            x[0] = 0
            # Inventory
            q = [0]*(N+1)
            q[0] = {'stake': 0, 'odds': 0}
            # # Optimal quotes
            rb = np.zeros((N))
            rl = np.zeros((N))
            # Order consumption probability factors
            M = 0.5
            A = 1./dt/math.exp(self.k*M/2)

            ### SIMULATION
            for n in range(N):
                ## Decide back and lay price
                rb[n], rl[n] = strategy.quotes(price=price[n])
                # Reserve deltas
                delta_b = rb[n] - price[n]
                delta_l = price[n] - rl[n]
                # Intensities
                lambda_b = A * math.exp(-self.k*delta_b)
                lambda_l = A * math.exp(-self.k*delta_l)
                # Order consumption (can be both per time step)
                yb = random.random()
                yl = random.random()
                ### Orders get filled or not?
                prob_back = 1 - math.exp(-lambda_b*dt)
                prob_lay = 1 - math.exp(-lambda_l*dt)
                dNb = 1 if yb < prob_back else 0
                dNl = 1 if yl < prob_lay else 0


                if q[n]=={'stake': 0, 'odds': 0}:
                    if (dNb - dNl)==0:
                        q[n+1] = q[n]
                    else:
                        q[n+1] = self.combine_bets(list_bets=[q[n],
                                                             {'stake': dNb, 'odds': rb[n]},
                                                             {'stake': -dNl, 'odds': rl[n]}])
                else:
                    if (q[n]['stake'] + dNb - dNl)==0:
                        dNb = 0
                        dNl = 0
                    q[n+1] = self.combine_bets(list_bets=[q[n],
                                                         {'stake': dNb, 'odds': rb[n]},
                                                         {'stake': -dNl, 'odds': rl[n]}])

                x[n+1] = x[n] - dNb + dNl
                pnl[n+1] = self.calculate_cash_out(stake=q[n+1]['stake'],
                                                   odds=q[n+1]['odds'],
                                                   current_odds=price[n])

            final_pnl[i_sim] = pnl[-1]

            pnl_series = pd.Series(pnl)
            returns = pnl_series.pct_change()
            returns = returns.replace(to_replace=[np.inf, np.NINF],
                                      value=np.nan)
            returns = returns.interpolate()
            returns = returns.replace(np.nan, 0)

            min_pnl[i_sim] = np.min(pnl)
            max_pnl[i_sim] = np.max(pnl)

            downside_std = np.nanstd(np.clip(returns, np.NINF, 0, out=None))
            mean_ret = np.mean(returns)
            std_returns = np.std(returns)
            mean_return[i_sim] = mean_ret
            volatility_returns[i_sim] = std_returns

            if mean_ret==0 or std_returns==0:
                sharpe_ratio[i_sim] = 0
            else:
                sharpe_ratio[i_sim] = mean_ret/std_returns


            if mean_ret==0 or downside_std==0:
                sortino_ratio[i_sim] = 0
            else:
                sortino_ratio[i_sim] = mean_ret/downside_std

            inv_stake = [inv['stake'] for inv in q]
            mean_inv_stake[i_sim] = np.mean(inv_stake)

        df_final_pnl = pd.DataFrame(final_pnl, columns=['Final PnL'])
        df_mean_return = pd.DataFrame(mean_return, columns=['Mean Return'])
        df_volatility = pd.DataFrame(volatility_returns, columns=['Volatility'])
        df_min_pnl = pd.DataFrame(min_pnl, columns=['Min PnL'])
        df_max_pnl = pd.DataFrame(max_pnl, columns=['Max PnL'])
        df_sharpe = pd.DataFrame(sharpe_ratio, columns=['Sharpe Ratio'])
        df_sortino = pd.DataFrame(sortino_ratio, columns=['Sortino Ratio'])
        df_mean_inv_stake = pd.DataFrame(mean_inv_stake, columns=['Mean Inv. Stake'])

        if plotting:
            print("\nResults over: %d simulations\n"%num_simulations)
            print("Final PnL")
            print(df_final_pnl.describe())
            print("Mean Return")
            print(df_mean_return.describe())
            print("Volatility")
            print(df_volatility.describe())
            print("Min PnL (Max Loss)")
            print(df_min_pnl.describe())
            print("Max PnL")
            print(df_max_pnl.describe())
            print("Sharpe ratio")
            print(df_sharpe.describe())
            print("Sortino Ratio")
            print(df_sortino.describe())
            print("Mean Inv. Stake")
            print(df_mean_inv_stake.describe())

            f = plt.figure(figsize=(15, 15))
            f.add_subplot(3, 2, 1)
            plt.plot(t, price, color='black', label='Mid price')
            plt.plot(t, rb, color='red',
                    label='Back price', markersize='4')
            plt.plot(t, rl, color='lime',
                    label='Lay price', markersize='2')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()

            f.add_subplot(3, 2, 2)
            plt.plot(t, pnl[:-1], color='black', label='P&L')
            plt.xlabel('Time')
            plt.ylabel('PnL [USD]')
            plt.grid(True)
            plt.legend()

            stake = [bet['stake'] for bet in q]
            f.add_subplot(3, 2, 3)
            plt.plot(t, stake[:-1], color='black', label='Total stake')
            plt.xlabel('Time')
            plt.ylabel('Inventory')
            plt.grid(True)
            plt.legend()

            odds = [bet['odds'] for bet in q]
            f.add_subplot(3, 2, 4)
            plt.plot(t, odds[:-1], color='black', label='Total odds')
            plt.xlabel('Time')
            plt.ylabel('Inventory')
            plt.grid(True)
            plt.legend()

            f.add_subplot(3, 2, 5)
            plt.plot(t, returns[:-1], color='black', label='Returns')
            plt.xlabel('Time')
            plt.ylabel('Returns')
            plt.grid(True)
            plt.legend()

            plt.savefig(os.path.join(plot_path, "last_simulation"))
            plt.close()

            for price in list_prices:
                plt.plot(price)
            plt.ylim(0.9, 100)
            plt.xlabel("Timesteps")
            plt.ylabel("Price (Odds)")
            plt.savefig(os.path.join(plot_path, "prices"))
            plt.close()

            bin_number = 30
            sns.histplot(final_pnl, bins=bin_number)
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "final_pnl"))
            plt.close()

            sns.histplot(mean_return, bins=bin_number)
            plt.xlabel('Mean Return')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "mean_return"))
            plt.close()

            sns.histplot(volatility_returns, bins=bin_number)
            plt.xlabel('Volatility')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "volatility"))
            plt.close()

            sns.histplot(min_pnl, bins=bin_number)
            plt.xlabel('Min PnL (Max Loss)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "min_pnl"))
            plt.close()

            sns.histplot(max_pnl, bins=bin_number)
            plt.xlabel('Max PnL')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "max_pnl"))
            plt.close()

            sns.histplot(sharpe_ratio, bins=bin_number)
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "sharpe_ratio"))
            plt.close()

            sns.histplot(sortino_ratio, bins=bin_number)
            plt.xlabel('Sortino ratio')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "sortino_ratio"))
            plt.close()

            sns.histplot(mean_inv_stake, bins=bin_number)
            plt.xlabel('Mean Inventory Stake')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_path, "mean_inv_stake"))
            plt.close()

            final_df_describe = pd.concat(objs=[df_final_pnl.describe(),
                                                df_mean_return.describe(),
                                                df_volatility.describe(),
                                                df_min_pnl.describe(),
                                                df_max_pnl.describe(),
                                                df_sharpe.describe(),
                                                df_sortino.describe(),
                                                df_mean_inv_stake.describe()],
                                          axis='columns')

            dfi.export(final_df_describe, os.path.join(plot_path, 'all_describes.png'))
            print(final_df_describe.to_latex())

        return {'final_pnl': final_pnl,
                'mean_return': mean_return,
                'volatility': volatility_returns,
                'min_pnl': min_pnl,
                'max_pnl': max_pnl,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'mean_inv_stake': mean_inv_stake,}


