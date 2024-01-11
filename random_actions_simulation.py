import matplotlib.pyplot as plt
from alive_progress import alive_it

from environments.gym_env import sportsTradingEnvironment
from utils import setup


def single_random_simulation(k: int, a_s: float, b_s: float) -> None:
    """
    Executes a single simulation (one match) of the sports trading environment
    (SportsTrading Environment class) using random actions and plots state variables
    (mid-price, momentum indicator and volatility indicator), PnL and inventory (stake
    and odds).

    Args:
        k (int): parameter of the AS framework, represnts the liquidity of the market (a
            higher value of k means lower liquidity).
        a_s (float): Has to be between 0 and 1. Represents the probability of player A winning
            a serving point.
        b_s (float): Has to be between 0 and 1. Represents the probability of player B winning
            a serving point.

    Returns: None
    """
    env = sportsTradingEnvironment.SportsTradingEnvironment(mode='fixed',
                                                            a_s=a_s,
                                                            b_s=b_s,
                                                            k=k)

    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action=action)

    f = plt.figure(figsize=(10, 10))
    f.add_subplot(2, 1, 1)
    plt.title("Environment simulation (with random actions)")
    plt.plot(env.price, label="Mid-price")
    plt.plot(env.list_momentum_indicator, label="Momentum indicator")
    plt.plot(env.list_volatility_indicator, label="Volatility indicator")
    plt.xlabel("Timesteps (points)")
    plt.legend()

    f.add_subplot(2, 1, 2)
    plt.plot(env.list_pnl, label="PnL")
    plt.plot(env.list_inventory_stake, label="Inventory stake")
    plt.plot(env.list_inventory_odds, label="Inventory odds")
    plt.xlabel("Timesteps (points)")
    plt.legend()
    plt.show()



if __name__=="__main__":
    _ = setup.setup()

    ## Environment parameters
    a_s = 0.7
    b_s = 0.7
    k = 10
    single_random_simulation(k=k, a_s=a_s, b_s=b_s)