import math
import random

import gymnasium as gym
#from gym import spaces
import numpy as np

from environments.tennis_simulator import tennisSimulator


class SportsTradingEnvironment(gym.Env):
    def __init__(self, a_s, b_s, k):
        super().__init__()
        self.a_s = a_s
        self.b_s = b_s

        self.action_space = gym.spaces.Discrete(100)

        ## normalized (inv.stake, inv.odd, price, momentum indicator)
        ### we have to NORMALIZE!!!!!!!!!
        self.observation_space = gym.spaces.Box(low=np.array([-50, -100, 1, -100]), high=np.array([50, 100, 100, 100]), dtype=np.float32)


        # Initialize state variables
        #self.mid_price = self.simulate_mid_price()
        # self.timestep = 0
        self.MOMENTUM_WINDOW_SIZE = 30
        self.STARTING_TIMESTEP = self.MOMENTUM_WINDOW_SIZE + 1


        self.timestep = self.STARTING_TIMESTEP
        self.q = {"stake": 0, "odds": 0}  # St: stake, Ot: odds
        self.x = 0
        self.pnl = 0
        self.momentum_indicator = 0
        self.price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=self.a_s, b_s=self.b_s)
        self.price = self.simulate_mid_price()
        self.max_timestep = len(self.price)
        self.back_prices = []
        self.lay_prices = []
        self.list_pnl = []
        self.list_inventory_stake = []
        self.list_inventory_odds = []
        self.back_offsets = []
        self.lay_offsets = []
        self.list_momentum_indicator = [0]*self.STARTING_TIMESTEP

        ### AS framework parameters
        self.k = k
        self.dt = 0.01
        self.M = 0.5
        self.A = 1./self.dt/math.exp(self.k*self.M/2)



    def simulate_mid_price(self):
        _, price = self.price_simulator.simulate()

        return price


    def combine_bets(self, list_bets):
        stake = sum([bet['stake'] for bet in list_bets])

        if stake==0:
            print("Total stake can't be 0")
            return False

        odds = sum([(bet['odds'] * (bet['stake']/stake)) for bet in list_bets])

        return {'stake': stake,
                'odds': odds}



    def calculate_cash_out(self, stake, odds, current_odds):
        current_stake = ((odds+1)*stake)/(current_odds+1)
        cash_out = (stake*odds) - (current_stake*current_odds)

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


    def update_inventory(self, dNb, dNl, rb, rl):
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



    def decode_action(self, action):
        # Decode the combined action into back and lay offsets
        back_offset = round((action // 10) * 0.1, 1)
        lay_offset = round((action % 10) * 0.1, 1)

        return back_offset, lay_offset


    def step(self, action):
        back_offset, lay_offset = self.decode_action(action)
        rb = self.price[self.timestep] + back_offset
        rl = self.price[self.timestep] - lay_offset

        # Simulate order book using Avellaneda-Stoikov framework and check if back and lay orders are filled
        dNb, dNl = self.avellaneda_stoikov_framework_step(rb=rb, rl=rl, price=self.price[self.timestep])
        self.update_inventory(dNb=dNb, dNl=dNl, rb=rb, rl=rl)

        ### Update Cash and PnL
        self.x = self.x - dNb + dNl
        self.pnl = self.calculate_cash_out(stake=self.q["stake"],
                                            odds=self.q["odds"],
                                            current_odds=self.price[self.timestep])

        ## momentum indicator
        self.momentum_indicator = self.price[self.timestep] - self.price[self.timestep-self.MOMENTUM_WINDOW_SIZE]


        # Move to the next timestep
        self.timestep += 1

        self.back_prices.append(rb)
        self.lay_prices.append(rl)
        self.back_offsets.append(back_offset),
        self.lay_offsets.append(lay_offset)
        self.list_inventory_stake.append(self.q['stake'])
        self.list_inventory_odds.append(self.q['odds'])
        self.list_pnl.append(self.pnl)
        self.list_momentum_indicator.append(self.momentum_indicator)

        # Return the state, reward, done, and any additional info
        return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator], dtype=np.float32), self.pnl, self.timestep >= self.max_timestep-1, False, {}


    def reset(self, seed=None):
        self.price = self.simulate_mid_price()
        self.max_timestep = len(self.price)
        self.timestep = self.STARTING_TIMESTEP
        self.momentum_indicator = 0
        self.q = {"stake": 0, "odds": 0}
        self.pnl = 0
        self.list_pnl = []
        self.back_prices = []
        self.lay_prices = []
        self.back_offsets = []
        self.lay_offsets = []
        self.list_inventory_odds = []
        self.list_inventory_stake = []
        self.list_momentum_indicator = [0]*self.STARTING_TIMESTEP

        return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator], dtype=np.float32), {}


    def render(self, mode='human'):
        print(f"Mid Price: {self.price[self.timestep]}, Time: {self.timestep}, Inventory: {self.q}, PnL: {self.pnl}")

