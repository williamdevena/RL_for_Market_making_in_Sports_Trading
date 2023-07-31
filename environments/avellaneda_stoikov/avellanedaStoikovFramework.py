import math
import random

#import brownian as bm
import matplotlib.pyplot as plt
import numpy

from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy

"""
High-frequency trading in a limit order book, Marco Avellaneda & Sasha Stoikov
paper url: https://www.researchgate.net/publication/24086205_High_Frequency_Trading_in_a_Limit_Order_Book

Some model limitations, discussed: https://quant.stackexchange.com/questions/36400/avellaneda-stoikov-market-making-model
Parameter fitting: https://quant.stackexchange.com/questions/36073/how-does-one-calibrate-lambda-in-a-avellaneda-stoikov-market-making-problem
"""


class AvellanedaStoikovFramework():

    def __init__(self, k=20.0, T=1.0):
        self.k = k
        self.T = T

    def run_simulation(self,
                       price_simulator,
                       strategy,
                       num_simulations=100):
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
        pnl_sim = numpy.empty((num_simulations))

        for i_sim in range(num_simulations):
            _, price, _ = price_simulator.simulate()
            price_simulator.restart()

            N = len(price)
            dt = self.T/N
            t = numpy.linspace(0.0, N*dt, N)

            # Wealth
            pnl = numpy.empty((N+1))
            pnl[0] = 0
            # Cash
            x = numpy.empty((N+2))
            x[0] = 0
            # Inventory
            q = numpy.empty((N+1))
            q[0] = 0
            # Reserve price
            r = numpy.empty((N))
            # Optimal quotes
            ra = numpy.empty((N))
            rb = numpy.empty((N))

            # Order consumption probability factors
            M = price[0]/200
            A = 1./dt/math.exp(self.k*M/2)

            max_q_held = 0
            min_q_held = 0

            ### SIMULATION
            for n in range(N):
                ### Core of simulation (where the strategy decides the quotes)
                if isinstance(strategy, AvellanedaStoikovStrategy):
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

            pnl_sim[i_sim] = pnl[-1]

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

        plt.show()

        print("\nResults over: %d simulations\n"%num_simulations)
        print("Average PnL: %.2f"% numpy.mean(pnl_sim))
        print("Standard deviation PnL: %.2f"% numpy.std(pnl_sim))

        range_min = int(min(pnl_sim) - abs(min(pnl_sim)))
        range_max = int(max(pnl_sim) + abs(min(pnl_sim)))
        plt.hist(pnl_sim,
        bins=30,
        #  range=(range_min, range_max)
        )
        plt.xlabel('PnL', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.show()











##########################################
#       Simulations
#########################################

        # Risk factor (->0: high risk, ->1: low risk)
        #gamma = 0.05
        # Market model
        #k = 20.0

        # The Wiener process parameter.
        #sigma = 2

        ## DEFINING QUOTES (CORE OF STRATEGIES)
                ## AVELLANEDA-STOIKOV
                # # Reserve price (for AS optimal quotes)
                # r[n] = price[n] - q[n] * gamma * sigma**2*(T-dt*n)
                # # Reserve spread
                # r_spread = 2 / gamma * math.log(1+gamma/k)
                # ra[n] = r[n] + r_spread/2
                # rb[n] = r[n] - r_spread/2

                # # ## FIXED OFFSET
                # # offset = 0.01
                # # ra[n] = s[n] + offset
                # # rb[n] = s[n] - offset

                # # ## RANDOM STRATEGY
                # # random_offset = random.random()
                # # ra[n] = s[n] + random_offset
                # # rb[n] = s[n] - random_offset



                # Option B: Unlimit time horizon
            # else:
            #     # Upper bound of inventory position
            #     w = 0.5 * gamma**2 * sigma**2 * (q_max+1)**2
            #     # Optimal quotes
            #     coef = gamma**2*sigma**2/(2*w-gamma**2*q[n]**2*sigma**2)

            #     ra[n] = price[n] + math.log(1+(1-2*q[n])*coef)/gamma
            #     rb[n] = price[n] + math.log(1+(-1-2*q[n])*coef)/gamma
            #     # Reserve price
            #     r[n] = (ra[n] + rb[n])/2





# def run_simulation(price_simulator,
#                    strategy,
#                    num_simulations=100,
#                    #sigma=2,
#                    #gamma=0.05,
#                    k=20.0):
#     """
#     Run simulations of a market-making strategy over a given price time series. The simulation results
#     are plotted and some statistical information about the performance of the strategy is printed.

#     Args:
#         price (array_like): A time series of the asset's price.
#         strategy (Object): An instance of a Strategy class which defines the quotes method.
#         num_simulations (int, optional): The number of simulations to run. Defaults to 100.
#         sigma (float, optional): Parameter for the Avellaneda-Stoikov (AS) model. Defaults to 2.
#         gamma (float, optional): Risk aversion parameter for the AS model. Defaults to 0.05.
#         k (float, optional): Market depth parameter. Parameter for the consumption rate (the bigger k
#             the least probable it is to get orders filled). Defaults to 20.0.

#     Returns:
#         None: This function doesn't return anything but generates graphs of the simulation results.
#     """
#     pnl_sim = numpy.empty((num_simulations))

#     for i_sim in range(num_simulations):

#         _, price, _ = price_simulator.simulate()
#         price_simulator.restart()

#         # print(len(price))
#         # print(price)
#         # Total time.
#         T = 1.0
#         N = len(price)
#         dt = T/N
#         t = numpy.linspace(0.0, N*dt, N)

#         # Wealth
#         pnl = numpy.empty((N+1))
#         pnl[0] = 0
#         # Cash
#         x = numpy.empty((N+2))
#         x[0] = 0
#         # Inventory
#         q = numpy.empty((N+1))
#         q[0] = 0
#         # Reserve price
#         r = numpy.empty((N))
#         # Optimal quotes
#         ra = numpy.empty((N))
#         rb = numpy.empty((N))

#         # Order consumption probability factors
#         M = price[0]/200
#         A = 1./dt/math.exp(k*M/2)

#         max_q_held = 0
#         min_q_held = 0

#         ### SIMULATION
#         for n in range(N):
#             ### Core of simulation (where the strategy decides the quotes)
#             if isinstance(strategy, AvellanedaStoikovStrategy):
#                 r[n], ra[n], rb[n] = strategy.quotes(price=price[n], remaining_time=T-(dt*n), k=k)
#             else:
#                 ra[n], rb[n] = strategy.quotes(price=price[n])

#             # Reserve deltas
#             delta_a = ra[n] - price[n]
#             delta_b = price[n] - rb[n]

#             # Intensities
#             lambda_a = A * math.exp(-k*delta_a)
#             lambda_b = A * math.exp(-k*delta_b)

#             # Order consumption (can be both per time step)
#             ya = random.random()
#             yb = random.random()

#             ### Orders get filled or not?
#             prob_ask = 1 - math.exp(-lambda_a*dt) # 1-exp(-lt) or just lt?
#             prob_bid = 1 - math.exp(-lambda_b*dt)
#             dNa = 1 if ya < prob_ask else 0
#             dNb = 1 if yb < prob_bid else 0

#             ## Update cash and inventory
#             q[n+1] = q[n] - dNa + dNb
#             x[n+1] = x[n] + ra[n]*dNa - rb[n]*dNb
#             pnl[n+1] = x[n+1] + q[n+1]*price[n]

#             if q[n+1] > max_q_held:
#                 max_q_held = q[n+1]
#             if q[n+1] < min_q_held:
#                 min_q_held = q[n+1]

#         pnl_sim[i_sim] = pnl[-1]

#     # print("Last simulation results:\n")
#     # print("Final inventory hold: ", q[-1])
#     # print("Last price: ", price[-1])
#     # print("Cash: ", x[-1])
#     # print("Final wealth: ", pnl[n+1])
#     # print("Max q held: ", max_q_held)
#     # print("Min q held: ", min_q_held)

#     f = plt.figure(figsize=(15, 4))
#     f.add_subplot(1, 3, 1)
#     plt.plot(t, price, color='black', label='Mid-market price')
#     plt.plot(t, r, color='blue',
#             #linestyle='dashed',
#             label='Reservation price')
#     plt.plot(t, ra, color='red',
#             #linestyle='', marker='.',
#             label='Price asked', markersize='4')
#     plt.plot(t, rb, color='lime',
#             #linestyle='', marker='o',
#             label='Price bid', markersize='2')
#     plt.xlabel('Time', fontsize=16)
#     plt.ylabel('Price [USD]', fontsize=16)
#     plt.grid(True)
#     plt.legend()
#     #plt.show()

#     f.add_subplot(1,3, 2)
#     plt.plot(t, pnl[:-1], color='black', label='P&L')
#     plt.xlabel('Time', fontsize=16)
#     plt.ylabel('PnL [USD]', fontsize=16)
#     plt.grid(True)
#     plt.legend()

#     f.add_subplot(1,3, 3)
#     plt.plot(t, q[:-1], color='black', label='Stocks held')
#     plt.xlabel('Time', fontsize=16)
#     plt.ylabel('Inventory', fontsize=16)
#     plt.grid(True)
#     plt.legend()

#     plt.show()

# #     print("\nParameters used in simulations:\n")

# #     #print("gamma: %.3f"%gamma)
# #     print("k: %.3f"%k)
# #     #print("sigma: %.3f"%sigma)
# #     print("T: %.2f"%T)
# #     print("n steps: %d"%N)
# #    # print("Using limited horizon:", limit_horizon)

#     print("\nResults over: %d simulations\n"%num_simulations)

#     print("Average PnL: %.2f"% numpy.mean(pnl_sim))
#     print("Standard deviation PnL: %.2f"% numpy.std(pnl_sim))

#     range_min = int(min(pnl_sim) - abs(min(pnl_sim)))
#     range_max = int(max(pnl_sim) + abs(min(pnl_sim)))
#     print(range_max, range_min)
#     plt.hist(pnl_sim,
#             bins=30,
#             #  range=(range_min, range_max)
#              )
#     plt.xlabel('PnL', fontsize=16)
#     plt.ylabel('Frequency', fontsize=16)
#     plt.show()