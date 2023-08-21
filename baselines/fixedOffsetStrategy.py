#from strategy.marketMakingStrategy import MarketMakingStrategy


class FixedOffsetStrategy(
    #MarketMakingStrategy
    ):
    """
    A MarketMakingStrategy subclass representing a market making strategy where the quotes are offset from the
    current price by a fixed amount. The same offset is added to the back price and subtracted from the lay price.

    Attributes:
        offset (float): The fixed offset from the current price for calculating back and lay prices.

    Methods:
        quotes(price): Computes the back (rb) and lay (rl) prices by adding and subtracting the fixed offset from
                       the current price.
    """

    def __init__(self, offset):
        super().__init__()
        self.offset = offset


    def quotes(self, price):
        """
        Computes the back (rb) and lay (rl) prices by adding and subtracting the fixed offset from the current price.

        Args:
            price (float): The current price.

        Returns:
            tuple: The back (rb) and lay (rl) prices.
        """
        rb = price + self.offset
        rl = price - self.offset

        ## quoted prices (odds) can't be lower than 1.0
        if rl<=1.0:
            rl = 1.01
        if rb<=1.0:
            rb = 1.01

        return rb, rl