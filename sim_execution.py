from livetrader import portfolio, execution
import typing as tp
import vectorbtpro as vbt
import queue as q
import pandas as pd
from livetrader.event import FillEvent, OrderEvent
from livetrader.logger_config import sim_flow_logger, flow_logger
import numpy as np


class SimulatedExecutionHandler(execution.ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage, or fill-ratio issues.

    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution handler.
    """

    def __init__(self, events: tp.Optional[q.Queue], data: pd.DataFrame, init_cash):
        self.events = events
        self.data = data
        self.acc = vbt.pf_enums.AccountState(cash=float(init_cash), position=0.0, debt=0.0, locked_cash=0.0,
                                             free_cash=float(init_cash))
        self.positions = []
        self.leverage = 4
        self.home_currency = "GBP_USD"
        self.balance = float(init_cash)

    def close_positions(self):
        """
        Close out all positions for a given symbol.
        """
        if self.acc.position == 0:
            return
        else:
            self.create_order(self.positions[0]["instrument"], self.acc.position)

    def get_last_close_price(self, currency_pair):
        return self.data.iloc[-1].Close

    def get_current_time(self):
        return self.data.iloc[-1].name

    def get_current_price(self, currency_pair, price_type="ASK"):
        if currency_pair == self.home_currency:
            return 1.0
        else:
            return self.data.iloc[-1].Close

    def get_account_summary(self):
        """
        Get account summary
        """
        if self.acc.position == 0:
            acc_summary = {"balance": self.balance, "openTradeCount": 0}
        elif self.acc.position > 0:
            acc_summary = {"balance": self.balance, "openTradeCount": 1}
        else:
            acc_summary = {"balance": self.balance, "openTradeCount": 1}
        return acc_summary

    def get_positions(self):
        """
        Get open positions
        """
        return self.positions

    def create_position(self, instrument, direction, averagePrice):
        if self.acc.position == 0:
            self.positions = []
        else:
            if direction == "LONG":
                position_dict = {"instrument": instrument,
                                 "long": {"units": abs(self.acc.position), "averagePrice": averagePrice},
                                 "short": {"units": 0.0}}
            else:
                position_dict = {"instrument": instrument,
                                 "long": {"units": 0.0},
                                 "short": {"units": self.acc.position, "averagePrice": averagePrice}}
            self.positions = [position_dict]

    @staticmethod
    def parse_order(event_direction, pf, current_price) -> float:
        order_result = 0
        if event_direction == 'BUY':
            order_result, _ = vbt.pf_nb.buy_nb(
                account_state=pf.acc,
                size=np.inf,
                price=current_price,
                leverage=pf.leverage,
                leverage_mode=vbt.pf_enums.LeverageMode.Eager,
                size_granularity=1,
                fees=0.00005 * pf.leverage,
            )
            order_result = order_result.size
        if event_direction == 'SELL':
            order_result, _ = vbt.pf_nb.sell_nb(
                account_state=pf.acc,
                size=np.inf,
                price=current_price,
                leverage=pf.leverage,
                size_granularity=1,
                fees=0.00005 * pf.leverage,
            )
            order_result = -order_result.size
        if event_direction == 'EXIT':
            if pf.acc.position != 0:
                order_result = -pf.acc.position
            else:
                order_result = np.nan
        return order_result

    def create_fill_event(self, filled, limit_order_size):
        if float(filled['units']) > 0:
            fill_event = FillEvent(timeindex=filled['time'],
                                   symbol=filled['instrument'], exchange='Oanda', quantity=float(filled['units']),
                                   direction="BUY", fill_cost=float(filled["price"]),
                                   commission=float(filled["halfSpreadCost"]),
                                   usd_exchange_rate=1, limit_order_size=limit_order_size)
            fill_event.execute_order_items = dict(prev_acc_state=filled['prev_acc_state'],
                                                  direction="BUY", units=filled['units'], price=filled['price'])
        else:

            fill_event = FillEvent(timeindex=filled['time'],
                                   symbol=filled['instrument'], exchange='Oanda',
                                   quantity=abs(float(filled['units'])),
                                   direction="SELL", fill_cost=float(filled["price"]),
                                   commission=float(filled["halfSpreadCost"]), usd_exchange_rate=1,
                                   limit_order_size=limit_order_size)
            fill_event.execute_order_items = dict(prev_acc_state=filled['prev_acc_state'],
                                                  direction="SELL", units=filled['units'], price=filled['price'])
        return fill_event

    def create_order(self, symbol, units, **kwargs):
        filled = None
        current_price = self.get_current_price(symbol)
        time = self.data.iloc[-1].name
        prev_acc_state = self.acc
        if units > 0:
            new_balance = None
            if self.acc.position + units >= 0:
                order_result, new_balance = vbt.pf_nb.short_buy_nb(
                    account_state=self.acc,
                    size=np.inf,
                    price=current_price,
                    size_granularity=1,
                    fees=0.00005 * self.leverage,
                )
            order_result, new_acc_state = vbt.pf_nb.buy_nb(
                account_state=self.acc,
                size=units,
                price=current_price,
                leverage=self.leverage,
                leverage_mode=vbt.pf_enums.LeverageMode.Eager,
                size_granularity=1,
                fees=0.00005 * self.leverage,
            )

            if order_result.size == units:
                self.acc = new_acc_state
                self.create_position(symbol, "LONG", current_price)
                filled = dict(time=time, type="ORDER_FILL", instrument=symbol, units=order_result.size,
                              price=current_price, halfSpreadCost=0.00005 * self.leverage,
                              prev_acc_state=prev_acc_state)
                if new_balance is not None:
                    flow_logger.info(f"New acc at 0: {new_balance}")
                    flow_logger.info(f"Balance changed from {self.balance} to {new_balance.free_cash}")
                    self.balance = new_balance.free_cash
        else:
            new_balance = None
            if self.acc.position + units <= 0:
                order_result, new_balance = vbt.pf_nb.long_sell_nb(
                    account_state=self.acc,
                    size=np.inf,
                    price=current_price,
                    size_granularity=1,
                    fees=0.00005 * self.leverage,
                )
            order_result, new_acc_state = vbt.pf_nb.sell_nb(
                account_state=self.acc,
                size=-units,
                price=current_price,
                leverage=self.leverage,
                size_granularity=1,
                fees=0.00005 * self.leverage,
            )
            if -order_result.size == units:
                self.acc = new_acc_state
                self.create_position(symbol, "SHORT", current_price)
                filled = dict(time=time, instrument=symbol, type="ORDER_FILL", units=-order_result.size,
                              price=current_price, halfSpreadCost=0.00005 * self.leverage,
                              prev_acc_state=prev_acc_state)
                if new_balance is not None:
                    flow_logger.info(f"New acc at 0: {new_balance}")
                    flow_logger.info(f"Balance changed from {self.balance} to {new_balance.free_cash}")
                    self.balance = new_balance.free_cash
        return filled

    def execute_order(self, event: OrderEvent, pf: tp.Optional[portfolio.Portfolio] = None) -> None:
        filled = None
        current_price = self.get_current_price(event.symbol)
        self.leverage = pf.leverage
        if event.type == 'ORDER':
            order_size = self.parse_order(event.direction, pf, current_price)
            if not np.isnan(order_size):
                filled = self.create_order(event.symbol, units=int(order_size))
        if filled is not None:
            sim_flow_logger.info(f"Order executed and filled: {filled}")
            fill_event = self.create_fill_event(filled, event.limit_order_size)
            self.events.put(fill_event)
        else:
            sim_flow_logger.warning(f"Order not executed: {event}")
            self.events.put(None)
