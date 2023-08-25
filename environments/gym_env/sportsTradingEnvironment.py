import copy
import math
import random

import gymnasium as gym
#from gym import spaces
import numpy as np

from environments.tennis_simulator import tennisSimulator


class SportsTradingEnvironment(gym.Env):
    def __init__(self,
                 mode='training',
                 a_s=None,
                 b_s=None,
                 k=None
                 ):
        super().__init__()
        self.tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

        if mode=='testing':
            if a_s==None or b_s==None or k==None:
                raise ValueError("When 'mode'=='testing' 'a_s', 'b_s' or 'k' can not be None")
        elif mode=='training':
            if not (a_s==None and b_s==None and k==None):
                raise ValueError("When 'mode'=='training' 'a_s', 'b_s' and 'k' have to be None (they are not used)")
        self.mode = mode

        if self.mode=='testing':
            self.a_s = a_s
            self.b_s = b_s
            self.k = k
        elif self.mode=='training':
            self.a_s = random.choice(self.tennis_probs)
            self.b_s = random.choice(self.tennis_probs)
            self.k = random.randint(3, 12)
        else:
            raise ValueError("Wrong 'mode' value")

        self.price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=self.a_s, b_s=self.b_s)

        self.action_space = gym.spaces.Discrete(100)

        ## normalized (inv.stake, inv.odd, price, momentum indicator, volatility indicator)
        ### we have to NORMALIZE!!!!!!!!!
        self.observation_space = gym.spaces.Box(low=np.array([-50, -100, 1, -100, -100]), high=np.array([50, 100, 100, 100, 100]), dtype=np.float32)


        # Initialize state variables
        #self.mid_price = self.simulate_mid_price()
        # self.timestep = 0
        self.MOMENTUM_WINDOW_SIZE = 15
        self.STARTING_TIMESTEP = self.MOMENTUM_WINDOW_SIZE + 1
        self.VOLATILITY_WINDOW_SIZE = 15

        self.timestep = self.STARTING_TIMESTEP
        self.q = {"stake": 0, "odds": 0}  # St: stake, Ot: odds
        self.x = 0
        self.pnl = 0
        self.momentum_indicator = 0
        self.volatility_indicator = 0
        self.price = self.simulate_mid_price()
        self.max_timestep = len(self.price)
        self.back_prices = [0]*self.STARTING_TIMESTEP
        self.lay_prices = [0]*self.STARTING_TIMESTEP
        self.list_pnl = [0]*self.STARTING_TIMESTEP
        self.list_inventory_stake = [0]*self.STARTING_TIMESTEP
        self.list_inventory_odds = [0]*self.STARTING_TIMESTEP
        self.back_offsets = [0]*self.STARTING_TIMESTEP
        self.lay_offsets = [0]*self.STARTING_TIMESTEP
        self.list_momentum_indicator = [0]*self.STARTING_TIMESTEP
        self.list_volatility_indicator = [0]*self.STARTING_TIMESTEP

        ### AS framework parameters
        #self.k = k

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
        previous_pnl = copy.deepcopy(self.pnl)
        self.pnl = self.calculate_cash_out(stake=self.q["stake"],
                                            odds=self.q["odds"],
                                            current_odds=self.price[self.timestep])
        pnl_return = self.pnl - previous_pnl

        ## momentum indicator
        self.momentum_indicator = self.price[self.timestep] - self.price[self.timestep-self.MOMENTUM_WINDOW_SIZE]

        ## volatility indicator
        prices_for_volatility = self.price[self.timestep-self.VOLATILITY_WINDOW_SIZE:self.timestep]
        self.volatility_indicator = np.std(prices_for_volatility)


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
        self.list_volatility_indicator.append(self.volatility_indicator)

        # print(f"Previous pnl: {previous_pnl}\nPnl: {self.pnl}\nReturn: {pnl_return}")
        # print("------------")

        # if self.timestep==(self.max_timestep-1):
        #     #print("finale")
        #     print(self.list_pnl)
        #     print(f"Sum: {sum(self.list_pnl)}")


        # if self.timestep==(self.max_timestep-1):
        #     reward = self.pnl
        # else:
        #     reward = pnl_return

        reward = self.pnl

        # Return the state, reward, done, and any additional info
        #return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator, self.volatility_indicator], dtype=np.float32), self.pnl, self.timestep >= self.max_timestep-1, False, {}
        return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator, self.volatility_indicator], dtype=np.float32), reward, self.timestep >= self.max_timestep-1, False, {}



    def reset(self, seed=None):

        if self.mode=='training':
            self.a_s = random.choice(self.tennis_probs)
            self.b_s = random.choice(self.tennis_probs)
            self.k = random.randint(3, 12)
            self.price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=self.a_s, b_s=self.b_s)
            self.A = 1./self.dt/math.exp(self.k*self.M/2)

        self.price = self.simulate_mid_price()
        self.max_timestep = len(self.price)
        self.timestep = self.STARTING_TIMESTEP
        self.momentum_indicator = 0
        self.volatility_indicator = 0
        self.q = {"stake": 0, "odds": 0}
        self.pnl = 0
        self.list_pnl = [0]*self.STARTING_TIMESTEP
        self.back_prices = [0]*self.STARTING_TIMESTEP
        self.lay_prices = [0]*self.STARTING_TIMESTEP
        self.back_offsets = [0]*self.STARTING_TIMESTEP
        self.lay_offsets = [0]*self.STARTING_TIMESTEP
        self.list_inventory_odds = [0]*self.STARTING_TIMESTEP
        self.list_inventory_stake = [0]*self.STARTING_TIMESTEP
        self.list_momentum_indicator = [0]*self.STARTING_TIMESTEP
        self.list_volatility_indicator = [0]*self.STARTING_TIMESTEP

        #print(self.k, self.a_s, self.b_s)

        return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator, self.volatility_indicator], dtype=np.float32), {}


    def render(self, mode='human'):
        print(f"Mid Price: {self.price[self.timestep]}, Time: {self.timestep}, Inventory: {self.q}, PnL: {self.pnl}")

