import random

from strategy.marketMakingStrategy import MarketMakingStrategy


class RandomStrategy(MarketMakingStrategy):
    """
    A MarketMakingStrategy subclass representing a market making strategy where the quotes are offset from the
    current price by a random amount. The same random amount is added to the ask price and subtracted from the bid price.

    Attributes:
        random_seed (int): The seed for the random number generator. It ensures that the random sequences
                           generated by the methods are reproducible.
        range_offset (tuple): The continuos range in which the random offset in choosen in.

    Methods:
        quotes(price): Computes the ask (ra) and bid (rb) prices by adding and subtracting a random offset from
                       the current price.
    """

    def __init__(self, range_offset):
        super().__init__()
        self.random_seed = 42
        random.seed(self.random_seed)

        if range_offset[0]>=range_offset[1] or range_offset[0]<0 or range_offset[1]<0:
            raise ValueError("range_offset has to have a value (a, b) where a,b>=0 and a<b")
        self.range_offset = range_offset

    def quotes(self, price):
        """
        Computes the ask (ra) and bid (rb) prices by adding and subtracting a random offset from the current price.

        Args:
            price (float): The current price.

        Returns:
            tuple: The ask (ra) and bid (rb) prices.
        """
        random_offset = random.uniform(self.range_offset[0], self.range_offset[1])
        ra = price + random_offset
        rb = price - random_offset

        return ra, rb




