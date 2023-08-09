import math
import os
import random

import dataframe_image as dfi
#import brownian as bm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy

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

    def run_single_simulation(self,
                       price_simulator,
                       strategy,
                       num_simulations=100,
                       plotting=False,
                       plot_path=None):
        """
        Run simulations of a market-making strategy over a given price time series. The simulation results
        are plotted and some statistical information about the performance of the strategy is printed.

        Args:
        price (array_like): A time series of the asset's price.
        strategy (Object): An instance of a Strategy class which defines the quotes method.
        num_simulations (int, optional): The number of simulations to run. Defaults to 100.
        sigma (float, optional): Parameter for the Avellaneda-Stoikov (AS) model. Defaults to 2.
        gamma (float, optional): Risk aversion parameter for the AS model. Defaults to 0.05.
        k (float, optional): Market depth parameter. Parameter for the consumption rate (the bigger k
        the least probable it is to get orders filled). Defaults to 20.0.

        Returns:
        None: This function doesn't return anything but generates graphs of the simulation results.
        """
        if plot_path and not os.path.exists(plot_path):
            os.makedirs(plot_path)

        final_pnl = np.zeros((num_simulations))
        volatility_pnl = np.zeros((num_simulations))
        min_pnl = np.zeros((num_simulations))
        max_pnl = np.zeros((num_simulations))

        list_prices = []

        for i_sim in range(num_simulations):
            _, price = price_simulator.simulate()
            #price_simulator.restart()
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
            q = np.zeros((N+1))
            q[0] = 0
            # Reserve price
            r = np.zeros((N))
            #print(r)
            # Optimal quotes
            ra = np.zeros((N))
            rb = np.zeros((N))

            # Order consumption probability factors
            #M = price[0]/2
            M = 0.5
            A = 1./dt/math.exp(self.k*M/2)
            #print(M, dt, A)

            max_q_held = 0
            min_q_held = 0

            ### SIMULATION
            for n in range(N):
                ### Core of simulation (where the strategy decides the quotes)
                if isinstance(strategy, AvellanedaStoikovStrategy):
                    #print("AS model")
                    r[n], ra[n], rb[n] = strategy.quotes(price=price[n],
                                                         remaining_time=self.T-(dt*n),
                                                         k=self.k)
                else:
                    ra[n], rb[n] = strategy.quotes(price=price[n])

                # Reserve deltas
                delta_a = ra[n] - price[n]
                delta_b = price[n] - rb[n]

                # Intensities
                lambda_a = A * math.exp(-self.k*delta_a)
                lambda_b = A * math.exp(-self.k*delta_b)

                # Order consumption (can be both per time step)
                ya = random.random()
                yb = random.random()

                ### Orders get filled or not?
                prob_ask = 1 - math.exp(-lambda_a*dt) # 1-exp(-lt) or just lt?
                prob_bid = 1 - math.exp(-lambda_b*dt)
                dNa = 1 if ya < prob_ask else 0
                dNb = 1 if yb < prob_bid else 0

                ## Update cash and inventory
                q[n+1] = q[n] - dNa + dNb
                x[n+1] = x[n] + ra[n]*dNa - rb[n]*dNb
                pnl[n+1] = x[n+1] + q[n+1]*price[n]

                if q[n+1] > max_q_held:
                    max_q_held = q[n+1]
                if q[n+1] < min_q_held:
                    min_q_held = q[n+1]

            final_pnl[i_sim] = pnl[-1]
            volatility_pnl[i_sim] = np.std(pnl)
            min_pnl[i_sim] = np.min(pnl)
            max_pnl[i_sim] = np.max(pnl)


        df_final_pnl = pd.DataFrame(final_pnl, columns=['Final PnL'])
        df_volatility = pd.DataFrame(volatility_pnl, columns=['Volatility'])
        df_min_pnl = pd.DataFrame(min_pnl, columns=['Min PnL'])
        df_max_pnl = pd.DataFrame(max_pnl, columns=['Max PnL'])

        if plotting:
            print("\nResults over: %d simulations\n"%num_simulations)
            print("Final PnL")
            print(df_final_pnl.describe())
            print("Volatility")
            print(df_volatility.describe())
            print("Min PnL (Max Loss)")
            print(df_min_pnl.describe())
            print("Max PnL")
            print(df_max_pnl.describe())

            f = plt.figure(figsize=(15, 4))
            f.add_subplot(1, 3, 1)
            plt.plot(t, price, color='black', label='Mid-market price')
            plt.plot(t, r, color='blue',
            #linestyle='dashed',
            label='Reservation price')
            plt.plot(t, ra, color='red',
            #linestyle='', marker='.',
            label='Price asked', markersize='4')
            plt.plot(t, rb, color='lime',
            #linestyle='', marker='o',
            label='Price bid', markersize='2')
            plt.xlabel('Time', fontsize=16)
            plt.ylabel('Price [USD]', fontsize=16)
            plt.grid(True)
            plt.legend()
            #plt.show()

            f.add_subplot(1,3, 2)
            plt.plot(t, pnl[:-1], color='black', label='P&L')
            plt.xlabel('Time', fontsize=16)
            plt.ylabel('PnL [USD]', fontsize=16)
            plt.grid(True)
            plt.legend()

            f.add_subplot(1,3, 3)
            plt.plot(t, q[:-1], color='black', label='Stocks held')
            plt.xlabel('Time', fontsize=16)
            plt.ylabel('Inventory', fontsize=16)
            plt.grid(True)
            plt.legend()

            plt.savefig(os.path.join(plot_path, "last_simulation"))
            #plt.show()
            plt.close()

            for price in list_prices:
                plt.plot(price)
            plt.ylim(0.9, 100)
            plt.xlabel("Timesteps")
            plt.ylabel("Price (Odds)")
            plt.savefig(os.path.join(plot_path, "prices"))
            #plt.show()
            plt.close()

            # range_min = int(min(pnl_sim) - abs(min(pnl_sim)))
            # range_max = int(max(pnl_sim) + abs(min(pnl_sim)))

            plt.hist(final_pnl,
            bins=50,
            #  range=(range_min, range_max)
            )
            plt.xlabel('PnL', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(os.path.join(plot_path, "final_pnl"))
            plt.close()
            dfi.export(df_final_pnl.describe(), os.path.join(plot_path, 'df_final_pnl.png'))

            plt.hist(volatility_pnl,
            bins=50,
            #  range=(range_min, range_max)
            )
            plt.xlabel('Volatility', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(os.path.join(plot_path, "volatility"))
            plt.close()
            dfi.export(df_volatility.describe(), os.path.join(plot_path, 'df_volatility.png'))

            plt.hist(min_pnl,
            bins=50,
            #  range=(range_min, range_max)
            )
            plt.xlabel('Min PnL (Max Loss)', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(os.path.join(plot_path, "min_pnl"))
            plt.close()
            dfi.export(df_min_pnl.describe(), os.path.join(plot_path, 'df_min_pnl.png'))

            plt.hist(max_pnl,
            bins=50,
            #  range=(range_min, range_max)
            )
            plt.xlabel('Max PnL', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(os.path.join(plot_path, "max_pnl"))
            plt.close()
            dfi.export(df_max_pnl.describe(), os.path.join(plot_path, 'df_max_pnl.png'))

            final_df_describe = pd.concat(objs=[df_final_pnl.describe(),
                                                df_volatility.describe(),
                                                df_min_pnl.describe(),
                                                df_max_pnl.describe()],
                                          axis='columns')

            dfi.export(final_df_describe, os.path.join(plot_path, 'all_describes.png'))
            print(final_df_describe.to_latex())


        return {'final_pnl':final_pnl,
                'volatility': volatility_pnl,
                'min_pnl': min_pnl,
                'max_pnl': max_pnl}

