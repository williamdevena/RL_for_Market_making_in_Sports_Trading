import math
import random

import gymnasium as gym
#from gym import spaces
import numpy as np

from environments.tennis_simulator import tennisSimulator


class SportsTradingEnvironment(gym.Env):
    def __init__(self, a_s, b_s, k):
        super(SportsTradingEnvironment, self).__init__()
        self.a_s = a_s
        self.b_s = b_s

        # Define action and observation space
        # Assuming the action space is continuous with back and lay prices ranging from some reasonable range (e.g., 1.01 to 100)
        self.action_space = gym.spaces.Box(low=1.01, high=100, shape=(2,), dtype=np.float32)

        # Observation: mid-price and timestep
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([100, 1000]), dtype=np.float32)

        # Initialize state variables
        #self.mid_price = self.simulate_mid_price()
        self.timestep = 0
        self.q = {"St": 0, "Ot": 0}  # St: stake, Ot: odds
        self.x = 0
        self.PnL = 0
        self.price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=self.a_s, b_s=self.b_s)
        self.price = self.simulate_mid_price()

        ### AS framework parameters
        self.k = k
        dt = 0.01
        self.M = 0.5
        self.A = 1./dt/math.exp(self.k*self.M/2)



    def simulate_mid_price(self):
        _, price = self.price_simulator.simulate()

        return price


    def calculate_cash_out(self, stake, odds, current_odds):
        current_stake = ((odds+1)*stake)/(current_odds+1)
        cash_out = (stake*odds) - (current_stake*current_odds)
        #print(cash_out)
        return cash_out


    def avellaneda_stoikov_framework_step(self, rb, rl, price):
        # Reserve deltas
        delta_b = rb - price
        delta_l = price - rl
        # Intensities
        lambda_b = self.A * math.exp(-self.k*delta_b)
        lambda_l = self.A * math.exp(-self.k*delta_l)
        # Order consumption (can be both per time step)
        yb = random.random()
        yl = random.random()
        ### Orders get filled or not?
        prob_back = 1 - math.exp(-lambda_b*self.dt) # 1-exp(-lt) or just lt?
        prob_lay = 1 - math.exp(-lambda_l*self.dt)
        dNb = 1 if yb < prob_back else 0
        dNl = 1 if yl < prob_lay else 0

        return dNb, dNl


    def step(self, action):
        # Action: [back_price, lay_price]
        rb, rl = action

        # Simulate order book using Avellaneda-Stoikov framework and check if back and lay orders are filled
        dNb, dNl = self.avellaneda_stoikov_framework_step(rb=rb, rl=rl, price=self.price)

        ## Update inventory
        if self.q=={'stake': 0, 'odds': 0}:
            if (dNb - dNl)==0:
                self.q = self.q
            else:
                self.q = self.combine_bets(list_bets=[self.q,
                                                        {'stake': dNb, 'odds': rb},
                                                        {'stake': -dNl, 'odds': rl}])
        else:
            if (self.q['stake'] + dNb - dNl)==0:
                dNb = 0
                dNl = 0
            self.q = self.combine_bets(list_bets=[self.q,
                                                    {'stake': dNb, 'odds': rb},
                                                    {'stake': -dNl, 'odds': rl}])

        ### Update Cash and PnL
        self.x = self.x - dNb + dNl
        self.PnL = self.calculate_cash_out(stake=self.q["stake"],
                                            odds=self.q["odds"],
                                            current_odds=self.price[self.timestep])

        # Move to the next timestep
        self.timestep += 1

        # Return the state, reward, done, and any additional info
        return [self.price, self.timestep], self.PnL, self.timestep >= 1000, {}

    def reset(self):
        self.price = self.simulate_mid_price()
        self.timestep = 0
        self.q = {"stake": 0, "odds": 0}
        self.PnL = 0
        return [self.price, self.timestep]

    def render(self, mode='human'):
        # For now, just print the current state. You can enhance this for better visualization later.
        print(f"Mid Price: {self.price}, Time: {self.timestep}, Inventory: {self.q}, PnL: {self.PnL}")

