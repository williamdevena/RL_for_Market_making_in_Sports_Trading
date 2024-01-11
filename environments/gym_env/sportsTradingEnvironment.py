import copy
import math
import random

import gymnasium as gym
import numpy as np

from environments.tennis_simulator import tennisSimulator
from typing import Optional, List, Tuple, Union, Dict


class SportsTradingEnvironment(gym.Env):
    """
    A custom Gymnasium environment for the simulation of a sports trading environment.
    It is used to train and test RL agents. As every class that inherits from gym.Env
    class, it has three main methods: 'step', 'reset' and 'render'. All the other methods
    are of support to these.

    Attributes:
        mode (str): The mode of the environment ('fixed' or 'random').
        a_s (float): Player A's probability of winning a point. Only used when mode is 'fixed'.
        b_s (float): Player B's probability of winning a point. Only used when mode is 'fixed'.
        k (int): Parameter for the Avellaneda-Stoikov framework. Only used when mode is 'fixed'.
        tennis_probs (list): List of possible probabilities for tennis players.
        action_space (gym.Space): The action space of the environment.
        observation_space (gym.Space): The observation space of the environment.
    """
    def __init__(self,
                 mode: str = 'random',
                 a_s: Optional[float] = None,
                 b_s: Optional[float] = None,
                 k: Optional[int] = None) -> None:
        """
        Initializes the SportsTradingEnvironment object.

        Args:
            mode (str): The mode of the environment ('fixed' or 'random'). It is used to specify
                if the environment has to be set on the 'fixed' mode, which means that the paramaters
                'k', 'a_s' and 'b_s' are specified and remain fixed during the execution, or 'random',
                which means that the environment randomly shuffles these three parameters during the
                execution.
            a_s (float, optional): Player A's probability of winning. Used by the Markov model, which
                simulates the price time series of an episode. If 'mode'='random' it has to be None.
            b_s (float, optional): Player B's probability of winning (same as 'a_s').
            k (int, optional): Parameter for the Avellaneda-Stoikov (AS) framework, which simulates
                the LOB environment. It represnts the liquidity of the market (a lower value means higher
                liquidity).

        Raises:
            ValueError: If the `mode` argument is not set correctly.

        """
        super().__init__()
        self.tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

        if mode=='fixed':
            if a_s==None or b_s==None or k==None:
                raise ValueError("When 'mode'=='fixed' 'a_s', 'b_s' or 'k' can not be None")
        elif mode=='random':
            if not (a_s==None and b_s==None and k==None):
                raise ValueError("When 'mode'=='random' 'a_s', 'b_s' and 'k' have to be None (they are not used)")
        self.mode = mode

        if self.mode=='fixed':
            self.a_s = a_s
            self.b_s = b_s
            self.k = k
        elif self.mode=='random':
            self.a_s = random.choice(self.tennis_probs)
            self.b_s = random.choice(self.tennis_probs)
            self.k = random.randint(3, 12)
        else:
            raise ValueError("Wrong 'mode' value")

        ## initialize the Markov model
        self.price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=self.a_s, b_s=self.b_s)
        ## Action space discrete from 0 to 100 (e.g. action 65 corresponds to back_offset=0.6 and lay_offset=0.5)
        self.action_space = gym.spaces.Discrete(100)
        ## inv.stake, inv.odd, price, momentum indicator, volatility indicator
        self.observation_space = gym.spaces.Box(low=np.array([-50, -100, 1, -100, -100]), high=np.array([50, 100, 100, 100, 100]), dtype=np.float32)


        # Initialize state variables
        self.MOMENTUM_WINDOW_SIZE = 15
        self.STARTING_TIMESTEP = self.MOMENTUM_WINDOW_SIZE + 1
        self.VOLATILITY_WINDOW_SIZE = 15
        self.timestep = self.STARTING_TIMESTEP
        self.q = {"stake": 0, "odds": 0}
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
        self.dt = 0.01
        self.M = 0.5
        self.A = 1./self.dt/math.exp(self.k*self.M/2)


    def simulate_mid_price(self) -> List[float]:
        """
        Simulates and returns the mid-price time series using the Tennis Markov model
        initialized in the init method.

        Returns:
            list: The simulated mid-price time series.

        """
        _, price = self.price_simulator.simulate()

        return price


    def combine_bets(self, list_bets: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combines multiple bets into a single bet with a new stake and odds. Used
        to calculate the current inventory (stake and odds) when placing new bets.

        Args:
            list_bets (list): List of bets to combine.

        Returns:
            dict: A dictionary containing the new 'stake' and 'odds'.

        """
        stake = sum([bet['stake'] for bet in list_bets])

        if stake==0:
            print("Total stake can't be 0")
            return False

        odds = sum([(bet['odds'] * (bet['stake']/stake)) for bet in list_bets])

        return {'stake': stake,
                'odds': odds}



    def calculate_cash_out(self, stake: float, odds: float, current_odds: float) -> float:
        """
        Calculates the cash-out value for a bet. Used to calculate
        the current value of an inventory (betting position).

        Args:
            stake (float): The stake of the bet.
            odds (float): The odds of the bet.
            current_odds (float): The current odds.

        Returns:
            float: The cash-out value.

        """
        current_stake = ((odds+1)*stake)/(current_odds+1)
        cash_out = (stake*odds) - (current_stake*current_odds)

        return cash_out


    def avellaneda_stoikov_framework_step(self, rb: float, rl: float, price: float) -> Tuple[int, int]:
        """
        Simulates a step in the Avellaneda-Stoikov (AS) framework.

        Args:
            rb (float): The back price.
            rl (float): The lay price.
            price (float): The current mid-price.

        Returns:
            tuple: A tuple containing the number of back and lay orders
                filled (0 or 1).

        """
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
        prob_back = 1 - math.exp(-lambda_b*self.dt)
        prob_lay = 1 - math.exp(-lambda_l*self.dt)
        dNb = 1 if yb < prob_back else 0
        dNl = 1 if yl < prob_lay else 0

        return dNb, dNl


    def update_inventory(self, dNb: int, dNl: int, rb: float, rl: float) -> None:
        """
        Updates the agent's inventory based on filled orders.

        Args:
            dNb (int): Number of back orders filled.
            dNl (int): Number of lay orders filled.
            rb (float): The back price.
            rl (float): The lay price.

        Returns: None

        """
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



    def decode_action(self, action: int) -> Tuple[float, float]:
        """
        Decodes a combined action into back and lay offsets. The
        actions have been defined as an integer between 0 and 100
        and one integer contains both the back and lay offset chosen
        by the agent (E.g. action=65 corresponds to back_offset=0.6
        and lay_offset=0.5).

        Args:
            action (int): The combined action.

        Returns:
            tuple: A tuple containing the back and lay offsets.

        """
        back_offset = round((action // 10) * 0.1, 1)
        lay_offset = round((action % 10) * 0.1, 1)

        return back_offset, lay_offset


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes one time step within the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            tuple: A tuple containing the new state, reward, done flag, and additional info.

        """
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
        reward = self.pnl

        # Return the state, reward, done, and any additional info
        return np.array([self.q['stake'], self.q['odds'], self.price[self.timestep], self.momentum_indicator, self.volatility_indicator], dtype=np.float32), reward, self.timestep >= self.max_timestep-1, False, {}



    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The seed for random number generation. Defaults to None.

        Returns:
            tuple: A tuple containing the initial state and additional info.

        """
        if self.mode=='random':
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

        return np.array([self.q['stake'],
                         self.q['odds'],
                         self.price[self.timestep],
                         self.momentum_indicator,
                         self.volatility_indicator], dtype=np.float32), {}


    def render(self, mode: str = 'human') -> None:
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode for rendering. Defaults to 'human'.

        """
        print(f"Mid Price: {self.price[self.timestep]}, Time: {self.timestep}, Inventory: {self.q}, PnL: {self.pnl}")

