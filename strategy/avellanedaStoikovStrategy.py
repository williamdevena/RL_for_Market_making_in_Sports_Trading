import math

from strategy.marketMakingStrategy import MarketMakingStrategy


class AvellanedaStoikovStrategy(MarketMakingStrategy):
    """
    A MarketMakingStrategy subclass representing the Avellaneda-Stoikov strategy for market making.
    This strategy sets ask (ra) and bid (rb) prices based on a calculated reservation price and reserve spread.

    Attributes:
        gamma (float): A parameter of the strategy related to risk aversion.
        sigma (float): A parameter of the strategy.

    Methods:
        quotes(price, remaining_time, k): Computes the ask (ra) and bid (rb) prices based on the Avellaneda-Stoikov
                                           strategy.
    """

    def __init__(self, gamma, sigma):
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma


    def quotes(self, price, remaining_time, k):
        """
        Computes the ask (ra) and bid (rb) prices based on the Avellaneda-Stoikov strategy. The quotes are
        calculated using a reservation price and a reserve spread.

        Args:
            price (float): The current price.
            remaining_time (float): The remaining time until the market closes.
            k (float): A parameter related to the orders' arrival intensity.

        Returns:
            tuple: The ask (ra) and bid (rb) prices.
        """
        reservation_price = price - self.q * self.gamma * self.sigma**2*(remaining_time)
        r_spread = 2 / self.gamma * math.log(1+self.gamma/k)
        ra = reservation_price + r_spread/2
        rb = reservation_price - r_spread/2

        return ra, rb