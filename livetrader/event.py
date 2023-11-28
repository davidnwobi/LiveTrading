import datetime


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """

    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with

    corresponding bars.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """

        self.type = "MARKET"

    def __str__(self):
        return "MarketEvent()"


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.

    This is received by a Portfolio object and acted upon.

    """

    def __init__(self, strategy_id: str, symbol: str, datetime: datetime, signal_type: str, strength: float = 1.0,
                 limit_order_size=0.001):
        """
        Initialises the SignalEvent.

        Parameters:
        - strategy_id: The unique identifier for the strategy that generated the signal.
        - symbol: The ticker symbol, e.g. 'GOOG'.
        - datetime: The timestamp at which the signal was generated.
        - signal_type: The type of signal, either 'LONG' or 'SHORT'.
        - strength: An adjustment factor used to scale the quantity at the portfolio level. Useful for pairs strategies.
        """

        self.type = "SIGNAL"
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.limit_order_size = limit_order_size
        self.debug_dict = {}

    def __str__(self) -> str:
        return f"SignalEvent(strategy_id={self.strategy_id}, symbol={self.symbol}, datetime={self.datetime}, " \
               f"signal_type={self.signal_type}, strength={self.strength})"


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity, and a direction.
    """

    def __init__(self, symbol: str, order_type: str, direction: str, date, limit_order_size: float = 0.001):
        """
        Initializes the order type, setting whether it is a Market order ('MKT') or Limit order ('LMT'),
        has a quantity (integral), and its direction ('BUY' or 'SELL').

        Parameters:
        - symbol: The instrument to trade.
        - order_type: 'MKT' or 'LMT' for Market or Limit.
        - quantity: Non-negative integer for quantity.
        - direction: 'BUY' or 'SELL' for long or short.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.limit_order_size = limit_order_size
        self.date = date

    def __str__(self) -> str:
        return f"OrderEvent(symbol={self.symbol}, order_type={self.order_type}, direction={self.direction}), {self.date}"

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print("Order: Symbol=%s, Type=%s, Direction=%s" %
              (self.symbol, self.order_type, self.direction))


# Limit Order Event: inherits from OrderEvent
class LimitOrderEvent(OrderEvent):
    """
    Handles the event of sending a Limit Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity, and a direction.
    """

    def __init__(self, symbol: str, order_type: str, direction: str, date, entry_price,
                 limit_price: float, limit_order_size: float = 0.001):
        """
        Initializes the order type, setting whether it is a Market order ('MKT') or Limit order ('LMT'),
        has a quantity (integral), and its direction ('BUY' or 'SELL').

        Parameters:
        - symbol: The instrument to trade.
        - order_type: 'MKT' or 'LMT' for Market or Limit.
        - quantity: Non-negative integer for quantity.
        - direction: 'BUY' or 'SELL' for long or short.
        """
        super().__init__(symbol, order_type, direction, date)
        self.type = 'LIMIT_ORDER'
        self.entry_price = entry_price
        self.limit_price = limit_price

    def __str__(self) -> str:
        return f"LimitOrderEvent(symbol={self.symbol}, order_type={self.order_type}, direction={self.direction}, " \
               f"limit_price={self.limit_price}) @ {self.date}"

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print("Order: Symbol=%s, Type=%s, Direction=%s, Limit Price=%s" %
              (self.symbol, self.order_type, self.direction, self.limit_price))


class FillEvent(Event):
    """
        Encapsulates the notion of a Filled Order, as returned from a brokerage.
        Stores the quantity of an instrument actually filled and at what price.
        In addition, stores the commission of the trade from the brokerage.
    """

    def __init__(self, timeindex: datetime, symbol: str, exchange: str, quantity: float, direction: str,
                 fill_cost: float,
                 commission: float = None, usd_exchange_rate=1, ask_price: float = None, avg_pip=1.5,
                 limit_order_size=1.0):
        """
        Initializes the FillEvent object. Sets the symbol, exchange, quantity, direction,
        cost of fill, and an optional commission.

        If commission is not provided, the Fill object will calculate it based on the trade size
        and Interactive Brokers fees.

        Parameters:
        - timeindex: The bar-resolution when the order was filled.
        - symbol: The instrument which was filled.
        - exchange: The exchange where the order was filled.
        - quantity: The filled quantity.
        - direction: The direction of fill ('BUY' or 'SELL').
        - fill_cost: The holdings value in dollars.
        - commission: An optional commission sent from Oanda.
        - ask_price: Used to calculate commission
        - bid_price: Used to calculate commission
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.ask_price = ask_price
        self.avg_pip = avg_pip
        self.usd_exchange_rate = usd_exchange_rate
        self.limit_order_size = limit_order_size
        self.execute_order_items = {}

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_oa_commission()
        else:
            self.commission = commission

    def __str__(self):
        return f"FillEvent: {self.symbol} {self.direction} {self.quantity} @ {self.fill_cost} GBP {self.commission} on {self.timeindex}"

    def calculate_oa_commission(self):
        """
        Calculates the fees of trading based on Oanda pricing + commission
        price sheet.
        https://www.oanda.com/assets/documents/566/OANDA-CC-Pricing.pdf
        """
        spread = 10 ** (len(str(self.ask_price).split('.')[0]) - 5) * self.avg_pip
        spread_cost = abs(spread) * self.quantity * 100000
        commission_cost = self.quantity * 5.0
        full_cost = spread_cost + commission_cost
        return full_cost
