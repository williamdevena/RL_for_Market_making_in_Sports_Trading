import numpy as np
from tennisim import match
from tennisim.sim import sim_game, sim_match, sim_set


class TennisMarkovSimulator():
    """
    This class simulates a tennis match using a Markov model.

    Attributes:
        a_s (float): Probability of player A winning a serve.
        b_s (float): Probability of player B winning a serve.

    Methods:
         simulate(): Simulates a tennis match and returns the match probabilities and odds time series.
    """

    def __init__(self, a_s, b_s):
        self.a_s = a_s
        self.b_s = b_s

    def simulate(self):
        """
        Simulates a tennis match and returns the match probabilities and odds time series.

        Note: The last probability is removed from the list to avoid a ZeroDivisionError when calculating odds.

        Returns:
            list_probs (list): The list of probabilities for the match, with the last probability removed.
            list_odds (list): The list of odds for the match, calculated as the reciprocal of each probability.
        """
        result_match = sim_match(a_s=self.a_s, b_s=self.b_s)
        formatted_result = match.reformat_match(match_data=result_match, p_a=self.a_s, p_b=self.b_s)
        list_probs = [x['prob'] for x in formatted_result]
        list_probs = list_probs[:-1]  # the last one could be 0 so we remove it to avoid the ZeroDivisionError
        list_odds = [1/prob for prob in list_probs]

        return list_probs, list_odds






