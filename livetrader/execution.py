import abc
import datetime

from .event import FillEvent, OrderEvent
from . import portfolio
import typing as tp


# Define the abstract base class ExecutionHandler
class ExecutionHandler(abc.ABC):
    """
    Abstract base class for handling order execution.
    """

    @abc.abstractmethod
    def get_account_summary(self):
        """
        Get account summary.

        Returns:
            dict: Account summary.
        """
        raise NotImplementedError("Should implement get_account_summary()")

    @abc.abstractmethod
    def get_positions(self):
        """
        Get open positions.

        Returns:
            dict: Open positions.
        """
        raise NotImplementedError("Should implement get_positions()")

    @abc.abstractmethod
    def create_order(self, instrument: str, units: int, max_retries: int = 5, retry_delay: int = 2, **kwargs) -> \
    tp.Optional[dict]:
        """
        Create an order with retry functionality.

        Parameters:
            instrument (str): Instrument to open the order for.
            units (int): Number of units to buy or sell.
            max_retries (int): Maximum number of retries in case of connection issues (default: 5).
            retry_delay (int): Delay in seconds between retries (default: 2).

        Returns:
            dict: Order response if successful, otherwise None.
        """
        raise NotImplementedError("Should implement create_order()")

    @abc.abstractmethod
    def get_current_price(self, currency_pair: str, price_type: str = "ASK") -> float:
        """
        Get the current price for a given currency pair.

        Parameters:
            currency_pair (str): Currency pair for which to get the price.
            price_type (str): Price type to fetch (default: "ASK").

        Returns:
            float: Current price.
        """
        raise NotImplementedError("Should implement get_current_price()")

    @abc.abstractmethod
    def get_current_time(self) -> datetime.datetime:
        """
        Get the current time.

        Returns:
            datetime.datetime: Current time.
        """
        raise NotImplementedError("Should implement get_current_time()")

    @abc.abstractmethod
    def close_positions(self):
        """
        Close all open positions.
        """
        raise NotImplementedError("Should implement close_positions()")

    @abc.abstractmethod
    def create_fill_event(self, filled: dict, limit_order_size: float) -> FillEvent:
        """
        Create a FillEvent object from the filled order details.

        Parameters:
            filled (dict): Filled order details.
            limit_order_size (float): Limit order size.

        Returns:
            FillEvent: A FillEvent object.
        """
        raise NotImplementedError("Should implement create_fill_event()")

    @abc.abstractmethod
    def execute_order(self, event: OrderEvent, pf: tp.Optional[portfolio.Portfolio] = None) -> None:
        """
        Execute the order based on the event.

        Parameters:
            event (OrderEvent): Contains an OrderEvent object with order information.
            pf (portfolio.Portfolio, optional): The Portfolio object.

        Returns:
            None
        """
        raise NotImplementedError("Should implement execute_order()")
