import matplotlib.pyplot as plt
from alive_progress import alive_it

from environments.tennis_simulator import tennisSimulator
from utils import setup


def price_simulations(a_s: float, b_s: float, num_simulations: int) -> None:
    """
    Simulates and plots the probability and odds of winning a tennis match over time.

    This function uses a Markov model to simulate the probability of winning and the odds for
    each player in a tennis match. It then plots these simulated values for a given number of simulations.

    Args:
        a_s (float): Player A's probability of winning a point.
        b_s (float): Player B's probability of winning a point.
        num_simulations (int): The number of simulations to run.

    Returns:
        None: This function only produces plots and does not return any values.

    """
    ## TENNIS MARKOV SIMULATOR
    simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)

    matches_probs_list = []
    matches_odds_list = []
    for x in alive_it(range(num_simulations)):
        prob_list, odds_list = simulator.simulate()
        matches_probs_list.append(prob_list)
        matches_odds_list.append(odds_list)

    f = plt.figure(figsize=(10, 10))
    f.add_subplot(2, 1, 1)
    for match_probs in matches_probs_list:
        plt.plot(match_probs)
    plt.title("Probability of winning the match")
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Probability")
    plt.xlabel("Timesteps (points)")

    f.add_subplot(2, 1, 2)
    for match_odds in matches_odds_list:
        plt.plot(match_odds)
    plt.title("Odds")
    plt.ylabel("Odds")
    plt.xlabel("Timesteps (points)")
    plt.show()



if __name__=="__main__":
    _ = setup.setup()

    ## Environment parameters
    a_s = 0.7
    b_s = 0.7
    num_simulations = 20
    price_simulations(a_s=a_s, b_s=b_s, num_simulations=num_simulations)