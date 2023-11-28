
import queue
from .event import SignalEvent, FillEvent, OrderEvent, LimitOrderEvent
import vectorbtpro as vbt
import typing as tp
from .logger_config import flow_logger, oandapyV20_logger


class Portfolio(object):
    """
    The Portfolio class handles the positions and market value of all instruments
    at a resolution of a "bar", i.e. secondly, minutely, 5-min, 30-min, 60-min, or EOD.
    The positions DataFrame stores a time-index of the quantity of positions held.
    The holdings DataFrame stores the cash and total market holdings value of each
    symbol for a particular time-index, as well as the percentage change in
    portfolio total across bars.
    """

    def __init__(self, events: queue.Queue, execution_handler):
        """

        """
        self.events = events
        self.execution_handler = execution_handler
        acc = execution_handler.get_account_summary()
        cash = float(acc["balance"])
        self.last_used_exchange_rate = execution_handler.get_current_price("GBP_USD")
        self.acc = vbt.pf_enums.AccountState(
            cash=cash * self.last_used_exchange_rate,
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=cash * self.last_used_exchange_rate
        )

        self.limit_order: tp.Optional[LimitOrderEvent] = None
        self.leverage = 5
        self.update_acc_positions(0.001)

    def get_last_close_price(self, currency_pair):
        return self.execution_handler.get_last_close_price(currency_pair)

    def get_current_time(self):
        return self.execution_handler.get_current_time()

    def update_limit_order_prices(self, current_price):
        if self.limit_order is not None:
            if self.acc.position > 0:
                if current_price > self.limit_order.entry_price:
                    limit_price = current_price * (1 - self.limit_order.limit_order_size)
                    self.limit_order.entry_price = current_price
                    self.limit_order.limit_price = limit_price
            else:
                if current_price < self.limit_order.entry_price:
                    limit_price = current_price * (1 + self.limit_order.limit_order_size)
                    self.limit_order.entry_price = current_price
                    self.limit_order.limit_price = limit_price

    def check_limit_order(self, current_price):
        if self.limit_order is not None:
            if self.limit_order.direction == "BUY" and self.limit_order.limit_price > current_price:
                order = OrderEvent(self.limit_order.symbol, "LMT", 'EXIT', date=self.get_current_time(),
                                   limit_order_size=self.limit_order.limit_order_size)
                return True, order
            elif self.limit_order.direction == "SELL" and self.limit_order.limit_price < current_price:
                order = OrderEvent(self.limit_order.symbol, "LMT", 'EXIT', date=self.get_current_time(),
                                   limit_order_size=self.limit_order.limit_order_size)
                return True, order

        return False, None

    @staticmethod
    def calculate_stop_loss(buying_price, max_percentage_loss, direction):
        if direction == "BUY":
            stop_loss_price = buying_price * (1 - max_percentage_loss)
        elif direction == "SELL":
            stop_loss_price = buying_price * (1 + max_percentage_loss)
        else:
            raise ValueError("Direction must be BUY or SELL")
        return stop_loss_price

    def update_acc_exchange(self):
        acc = self.execution_handler.get_account_summary()
        cash = float(acc["balance"])

        if self.acc.position != 0:
            current_exchange_rate = self.execution_handler.get_current_price("GBP_USD")
            conversion = current_exchange_rate / self.last_used_exchange_rate
            self.acc = vbt.pf_enums.AccountState(
                cash=self.acc.cash * conversion,
                position=self.acc.position,
                debt=self.acc.debt * conversion,
                locked_cash=self.acc.locked_cash * conversion,
                free_cash=self.acc.free_cash * conversion
            )
            self.last_used_exchange_rate = current_exchange_rate
        else:
            current_exchange_rate = self.execution_handler.get_current_price("GBP_USD")
            self.acc = vbt.pf_enums.AccountState(
                cash=cash * current_exchange_rate,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=cash * current_exchange_rate
            )
            self.last_used_exchange_rate = current_exchange_rate

    def update_acc_buy(self, fill: FillEvent, acc, positions):
        if len(positions) == 0:
            self.last_used_exchange_rate = self.execution_handler.get_current_price("GBP_USD")
            self.acc = vbt.pf_enums.AccountState(
                cash=float(acc["balance"]) * self.last_used_exchange_rate,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=float(acc["balance"]) * self.last_used_exchange_rate
            )
        else:
            self.last_used_exchange_rate = fill.usd_exchange_rate
            cash = -float(positions[0]['long']['units']) * float(positions[0]['long']['averagePrice']) * (
                    1 - 1 / self.leverage)
            position = float(positions[0]['long']['units'])
            debt = float(positions[0]['long']['units']) * float(positions[0]['long']['averagePrice']) * (
                    1 - 1 / self.leverage)
            locked_cash = float(positions[0]['long']['units']) * float(
                positions[0]['long']['averagePrice']) * (
                                  1 / self.leverage)
            free_cash = 0
            self.acc = vbt.pf_enums.AccountState(
                cash=cash,
                position=position,
                debt=debt,
                locked_cash=locked_cash,
                free_cash=0.0
            )

    def update_acc_sell(self, fill: FillEvent, acc, positions):

        self.last_used_exchange_rate = self.execution_handler.get_current_price("GBP_USD")
        if len(positions) == 0:
            self.acc = vbt.pf_enums.AccountState(
                cash=float(acc["balance"]) * self.last_used_exchange_rate,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=float(acc["balance"]) * self.last_used_exchange_rate
            )
        else:
            self.last_used_exchange_rate = fill.usd_exchange_rate
            cash = float(acc["balance"]) * fill.usd_exchange_rate - float(positions[0]['short']['units']) * \
                   float(positions[0]['short']['averagePrice'])
            debt = abs(float(positions[0]['short']['units'])) * float(positions[0]['short']['averagePrice'])
            locked_cash = abs(float(positions[0]['short']['units'])) * \
                          float(positions[0]['short']['averagePrice']) * \
                          (1 / self.leverage)
            position = float(positions[0]['short']['units'])
            self.acc = vbt.pf_enums.AccountState(
                cash=cash,
                position=position,
                debt=debt,
                locked_cash=locked_cash,
                free_cash=cash - debt - locked_cash)

    def update_limit_order(self, positions, fill: FillEvent):
        if self.acc.position != 0:
            if fill.direction == 'BUY':
                limit_price = self.calculate_stop_loss(float(positions[0]['long']['averagePrice']),
                                                       fill.limit_order_size, fill.direction)
                self.limit_order = LimitOrderEvent(symbol=positions[0]["instrument"], order_type="LMT",
                                                   direction="BUY", limit_price=limit_price,
                                                   entry_price=float(positions[0]['long']['averagePrice']),
                                                   date=self.get_current_time(),
                                                   limit_order_size=fill.limit_order_size)
            elif fill.direction == 'SELL':
                limit_price = self.calculate_stop_loss(float(positions[0]['short']['averagePrice']),
                                                       fill.limit_order_size, fill.direction)
                self.limit_order = LimitOrderEvent(symbol=positions[0]["instrument"], order_type="LMT",
                                                   direction="SELL", limit_price=limit_price,
                                                   entry_price=float(positions[0]['short']['averagePrice']),
                                                   date=self.get_current_time(),
                                                   limit_order_size=fill.limit_order_size)
        else:
            self.limit_order = None

    def update_positions_from_fill(self, fill: FillEvent):
        """
        Takes a Fill object and updates the position matrix to reflect the new position.

        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        if fill:
            acc = self.execution_handler.get_account_summary()
            positions = self.execution_handler.get_positions()
            flow_logger.info("Real acc summary: " + str(acc))
            flow_logger.info("Positions: " + str(positions))

            if fill.direction == 'BUY':
                self.update_acc_buy(fill, acc, positions)
                self.update_limit_order(positions, fill)

            if fill.direction == 'SELL':
                self.update_acc_sell(fill, acc, positions)
                self.update_limit_order(positions, fill)
            flow_logger.info("Order Filled: " + str(fill))
            flow_logger.info("Portfolio Updated")
            flow_logger.info("Portfolio Account state: " + str(self.acc))
            flow_logger.info("Portfolio Limit order: " + str(self.limit_order))

    def update_acc_positions(self, limit_order_size):
        acc = self.execution_handler.get_account_summary()
        positions = self.execution_handler.get_positions()

        fill = FillEvent(timeindex=self.get_current_time(), symbol="", exchange='Oanda',
                         quantity=float("nan"), direction="", fill_cost=float("nan"), commission=float("nan"),
                         usd_exchange_rate=self.execution_handler.get_current_price("GBP_USD"),
                         limit_order_size=limit_order_size)
        self.last_used_exchange_rate = fill.usd_exchange_rate
        if len(positions) == 0:
            self.acc = vbt.pf_enums.AccountState(
                cash=float(acc["balance"]) * self.last_used_exchange_rate,
                position=0.0,
                debt=0.0,
                locked_cash=0.0,
                free_cash=float(acc["balance"]) * self.last_used_exchange_rate
            )
        elif len(positions) == 1:
            direction = 'BUY' if float(positions[0]['long']['units']) > 0 else 'SELL'
            fill.direction = direction
            if direction == 'BUY':
                self.update_acc_buy(fill, acc, positions)
                self.update_limit_order(positions, fill)
            elif direction == 'SELL':
                self.update_acc_sell(fill, acc, positions)
                self.update_limit_order(positions, fill)
            flow_logger.info("Portfolio Updated")
            flow_logger.info("Portfolio Account state: " + str(self.acc))
            flow_logger.info("Portfolio Limit order: " + str(self.limit_order))
        else:
            flow_logger.error("More than one position in the portfolio. Not supported yet")
            raise ValueError("More than one position in the portfolio. Not supported yet")

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings from a FillEvent.
        """
        if event and event.type == 'FILL':
            self.update_positions_from_fill(event)
        else:
            flow_logger.info("No fill event to update portfolio")

    def update_signal(self, event, last_close):
        """
        Acts on a SignalEvent to generate new orders based on the portfolio logic.
        """
        if event and event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event, last_close)
            if order_event is not None:
                self.events.put(order_event)
        else:
            flow_logger.info("No signal event to update portfolio")

    def generate_naive_order(self, event: SignalEvent, last_close):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The tuple containing Signal information.

        Returns:
        order - The Order object to execute.
        """

        limit_triggered, order = self.check_limit_order(last_close)
        symbol = event.symbol
        direction = event.signal_type
        strength = event.strength
        self.update_acc_exchange()
        order_type = 'MKT'
        if limit_triggered:
            flow_logger.info("Limit order triggered")
            if self.acc.position > 0 and direction == 'SHORT':
                order = OrderEvent(symbol, order_type, 'SELL', date=event.datetime,
                                   limit_order_size=event.limit_order_size)
            elif self.acc.position < 0 and direction == 'LONG':
                order = OrderEvent(symbol, order_type, 'BUY', date=event.datetime,
                                   limit_order_size=event.limit_order_size)
        else:
            if direction == 'LONG':
                order = OrderEvent(symbol, order_type, 'BUY', date=event.datetime,
                                   limit_order_size=event.limit_order_size)
            if direction == 'SHORT':
                order = OrderEvent(symbol, order_type, 'SELL', date=event.datetime,
                                   limit_order_size=event.limit_order_size)
            if direction == 'LONG_EXIT':
                if not self.acc.position <= 0:
                    order = OrderEvent(symbol, order_type, 'EXIT', date=event.datetime,
                                       limit_order_size=event.limit_order_size)
            if direction == 'SHORT_EXIT':
                if not self.acc.position >= 0:
                    order = OrderEvent(symbol, order_type, 'EXIT', date=event.datetime,
                                       limit_order_size=event.limit_order_size)
        if order is not None:
            flow_logger.info("Order Created: " + str(order))
        else:
            flow_logger.info("No Order Created")
        return order
