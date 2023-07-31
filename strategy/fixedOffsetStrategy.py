from strategy.marketMakingStrategy import MarketMakingStrategy


class FixedOffsetStrategy(MarketMakingStrategy):
    """
    A MarketMakingStrategy subclass representing a market making strategy where the quotes are offset from the
    current price by a fixed amount. The same offset is added to the ask price and subtracted from the bid price.

    Attributes:
        offset (float): The fixed offset from the current price for calculating ask and bid prices.

    Methods:
        quotes(price): Computes the ask (ra) and bid (rb) prices by adding and subtracting the fixed offset from
                       the current price.
    """

    def __init__(self, offset):
        super().__init__()
        self.offset = offset


    def quotes(self, price):
        """
        Computes the ask (ra) and bid (rb) prices by adding and subtracting the fixed offset from the current price.

        Args:
            price (float): The current price.

        Returns:
            tuple: The ask (ra) and bid (rb) prices.
        """
        ra = price + self.offset
        rb = price - self.offset

        return ra, rb