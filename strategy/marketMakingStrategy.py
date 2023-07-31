from abc import ABC, abstractmethod


class MarketMakingStrategy(ABC):

    @abstractmethod
    def quotes(self, price):
        pass
