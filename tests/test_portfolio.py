import unittest
import tpqoa
import datetime
from livetrader.oanda_excecution import OandaExecutionHandler
from livetrader.event import SignalEvent, FillEvent, OrderEvent, LimitOrderEvent
import queue as q

from datetime import datetime
from livetrader.portfolio import Portfolio
import vectorbtpro as vbt
from pathlib import Path
from sim_execution import SimulatedExecutionHandler
import time
import tracemalloc
import numpy as np
import datetime as dt
import warnings

path = Path('D:/Data')
currency = 'SPX500_USD'
data = vbt.HDFData.fetch(path / currency / (currency + "_" + "A.hdf") / currency).get()
oanda = tpqoa.tpqoa("../oanda.cfg")


class TestSimExecutionHandler(unittest.TestCase):
    symbol = "SPX500_USD"
    limit_order_size = 0.0001
    buy_order = OrderEvent(symbol, "MKT", 'BUY', date=dt.datetime.now(), limit_order_size=limit_order_size)
    sell_order = OrderEvent(symbol, "MKT", 'SELL', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    exit_order = OrderEvent(symbol, "MKT", 'EXIT', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    buy_filled = {'time': '2023-07-19T13:59:17.864669042Z', 'instrument': symbol, 'units': '715517.0',
                  'price': 0.67812, 'halfSpreadCost': '38.8158'}
    sell_filled = {'time': '2023-07-19T13:59:17.864669042Z', 'instrument': symbol, 'units': '-715517.0',
                   'price': 0.67812, 'halfSpreadCost': '38.8158'}

    @classmethod
    def setUpClass(cls):
        tracemalloc.start()
        warnings.simplefilter("ignore", ResourceWarning)

    def setUp(self):
        self.local_events = q.Queue()
        self.local_oa_execution_handler = OandaExecutionHandler(self.local_events, data)
        self.symbol = "SPX500_USD"
        self.local_oa_execution_handler.close_positions()

    def test_oanda_execution_handler_close_positions(self):
        time.sleep(2)
        acc_sum = self.local_oa_execution_handler.get_account_summary()
        self.assertEqual(acc_sum['openTradeCount'], 0)

    def test_oanda_execution_handler_get_current_price(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        self.assertIsNotNone(current_price)

    def test_oanda_execution_handler_parse_order(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        local_events = q.Queue()
        local_portfolio = Portfolio(local_events, self.local_oa_execution_handler)  # position = 0
        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)

        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        local_portfolio.acc = vbt.pf_enums.AccountState(cash=-388680.817164326, position=717258.0,
                                                        debt=388680.817164326, locked_cash=97170.2042910815,
                                                        free_cash=0.0)  # on a long position

        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        local_portfolio.acc = vbt.pf_enums.AccountState(cash=584603.8795027883, position=-719067.0,
                                                        debt=487145.96827415033, locked_cash=97429.19365483007,
                                                        free_cash=0.0)
        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

    def test_oanda_execution_handler_create_fill_event(self):
        local_events = q.Queue()
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # position = 0
        fill_event = self.local_oa_execution_handler.create_fill_event(self.buy_filled, 0.0001)

        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.direction, "BUY")
        self.assertEqual(fill_event.fill_cost, float(self.buy_filled['price']))
        self.assertEqual(fill_event.quantity, float(self.buy_filled['units']))

        fill_event = self.local_oa_execution_handler.create_fill_event(self.sell_filled, 0.0001)
        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.direction, "SELL")
        self.assertEqual(fill_event.fill_cost, float(self.sell_filled['price']))
        self.assertEqual(fill_event.quantity, abs(float(self.sell_filled['units'])))

    def test_oanda_execution_handler_create_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, 1)

        self.assertEqual(order['type'], 'ORDER_FILL')
        self.assertEqual(float(order['units']), 1.0)

        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, -2)
        self.assertEqual(order['type'], 'ORDER_FILL')
        self.assertEqual(float(order['units']), -2.0)

        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, -2000 * 2000 * 2000 * 2000)
        self.assertIsNone(order)

    def test_oanda_execution_handler_execute_order(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # position = 0

        # buy order
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "BUY")
        local_portfolio.update_fill(filled_order)

        # exit buy order
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "SELL")

        # sell order
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # reset portfolio
        self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "SELL")
        local_portfolio.update_fill(filled_order)

        # exit sell order
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "BUY")

    @classmethod
    def tearDownClass(cls) -> None:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)


class TestOandaExecutionHandler(unittest.TestCase):
    symbol = "AUD_USD"
    limit_order_size = 0.0001
    buy_order = OrderEvent(symbol, "MKT", 'BUY', date=dt.datetime.now(), limit_order_size=limit_order_size)
    sell_order = OrderEvent(symbol, "MKT", 'SELL', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    exit_order = OrderEvent(symbol, "MKT", 'EXIT', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    buy_filled = {'time': '2023-07-19T13:59:17.864669042Z', 'instrument': symbol, 'units': '715517.0',
                  'price': 0.67812, 'halfSpreadCost': '38.8158'}
    sell_filled = {'time': '2023-07-19T13:59:17.864669042Z', 'instrument': symbol, 'units': '-715517.0',
                   'price': 0.67812, 'halfSpreadCost': '38.8158'}

    @classmethod
    def setUpClass(cls):
        tracemalloc.start()
        warnings.simplefilter("ignore", ResourceWarning)

    def setUp(self):
        self.local_events = q.Queue()
        self.local_oa_execution_handler = OandaExecutionHandler(self.local_events)
        self.symbol = "AUD_USD"
        self.local_oa_execution_handler.close_positions()

    def test_oanda_execution_handler_close_positions(self):
        time.sleep(2)
        acc_sum = self.local_oa_execution_handler.get_account_summary()
        self.assertEqual(acc_sum['openTradeCount'], 0)

    def test_oanda_execution_handler_get_current_price(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        self.assertIsNotNone(current_price)

    def test_oanda_execution_handler_parse_order(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        local_events = q.Queue()
        local_portfolio = Portfolio(local_events, self.local_oa_execution_handler)  # position = 0
        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)

        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        local_portfolio.acc = vbt.pf_enums.AccountState(cash=-388680.817164326, position=717258.0,
                                                        debt=388680.817164326, locked_cash=97170.2042910815,
                                                        free_cash=0.0)  # on a long position

        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertLess(order_result, 0)

        local_portfolio.acc = vbt.pf_enums.AccountState(cash=584603.8795027883, position=-719067.0,
                                                        debt=487145.96827415033, locked_cash=97429.19365483007,
                                                        free_cash=0.0)
        order_result = OandaExecutionHandler.parse_order(self.sell_order.direction, local_portfolio, current_price)
        self.assertTrue(np.isnan(order_result))

        order_result = OandaExecutionHandler.parse_order(self.exit_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

        order_result = OandaExecutionHandler.parse_order(self.buy_order.direction, local_portfolio, current_price)
        self.assertFalse(np.isnan(order_result))
        self.assertGreater(order_result, 0)

    def test_oanda_execution_handler_create_fill_event(self):
        local_events = q.Queue()
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # position = 0
        fill_event = self.local_oa_execution_handler.create_fill_event(self.buy_filled, 0.0001)

        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.direction, "BUY")
        self.assertEqual(fill_event.fill_cost, float(self.buy_filled['price']))
        self.assertEqual(fill_event.quantity, float(self.buy_filled['units']))

        fill_event = self.local_oa_execution_handler.create_fill_event(self.sell_filled, 0.0001)
        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.direction, "SELL")
        self.assertEqual(fill_event.fill_cost, float(self.sell_filled['price']))
        self.assertEqual(fill_event.quantity, abs(float(self.sell_filled['units'])))

    def test_oanda_execution_handler_create_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, 1)
        self.assertEqual(order['type'], 'ORDER_FILL')
        self.assertEqual(float(order['units']), 1.0)

        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, -2)
        self.assertEqual(order['type'], 'ORDER_FILL')
        self.assertEqual(float(order['units']), -2.0)

        time.sleep(2)
        order = self.local_oa_execution_handler.create_order(self.symbol, -2000 * 2000 * 2000 * 2000)
        self.assertIsNone(order)

    def test_oanda_execution_handler_execute_order(self):
        current_price = self.local_oa_execution_handler.get_current_price(self.symbol)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # position = 0

        # buy order
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "BUY")
        local_portfolio.update_fill(filled_order)

        # exit buy order
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "SELL")

        # sell order
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)  # reset portfolio
        self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "SELL")
        local_portfolio.update_fill(filled_order)

        # exit sell order
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)
        self.assertFalse(self.local_events.empty())
        filled_order = self.local_events.get()
        self.assertIsInstance(filled_order, FillEvent)
        self.assertEqual(filled_order.direction, "BUY")

    @classmethod
    def tearDownClass(cls) -> None:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)


class TestPortfolioSim(unittest.TestCase):
    symbol = currency
    limit_order_size = 0.0001
    buy_order = OrderEvent(symbol, "MKT", 'BUY', date=dt.datetime.now(), limit_order_size=limit_order_size)
    sell_order = OrderEvent(symbol, "MKT", 'SELL', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    exit_order = OrderEvent(symbol, "MKT", 'EXIT', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)

    def setUp(self) -> None:
        warnings.simplefilter("ignore", ResourceWarning)
        OandaExecutionHandler.close_positions()
        self.local_events = q.Queue()
        self.local_oa_execution_handler = SimulatedExecutionHandler(self.local_events, data, init_cash=100000)
        self.symbol = currency

    def test_portfolio_init_updates_positions(self):

        oanda_position = self.local_oa_execution_handler.get_positions()
        acc_summary = self.local_oa_execution_handler.get_account_summary()
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)

        self.assertEqual(local_portfolio.acc.position, 0.0)
        self.assertAlmostEqual(local_portfolio.acc.cash,
                               (float(acc_summary['balance']) *
                                local_portfolio.last_used_exchange_rate), delta=10)

        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        fill = self.local_events.get()
        local_portfolio.update_fill(fill)
        prev_acc = local_portfolio.acc
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.assertEqual(local_portfolio.acc.position, prev_acc.position)
        self.assertAlmostEqual(local_portfolio.acc.cash, prev_acc.cash, delta=10)

        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)
        fill = self.local_events.get()
        local_portfolio.update_fill(fill)

        prev_acc = local_portfolio.acc
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.assertEqual(local_portfolio.acc.position, prev_acc.position)
        self.assertAlmostEqual(local_portfolio.acc.cash, prev_acc.cash, delta=10)

    def test_portfolio_get_last_close_price(self):
        last_close_price = self.local_oa_execution_handler.get_last_close_price(self.symbol)
        self.assertIsNotNone(last_close_price)

    def test_portfolio_update_signal_returns_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        try:
            order = self.local_events.get()
        except q.Empty:
            order = None
        self.assertIsNotNone(order)

    def testt_portfolio_update_signal_into_position(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "BUY")

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "SELL")

    def test_portfolio_update_signal_out_of_position(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        self.assertIs(self.local_events.empty(), True)

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertIs(self.local_events.empty(), True)
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "EXIT")

        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()

        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "EXIT")

    def test_portfolio_update_acc_buy_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        self.assertFalse(self.local_events.empty())

        fill = self.local_events.get()
        acc = self.local_oa_execution_handler.get_account_summary()
        positions = self.local_oa_execution_handler.get_positions()

        local_portfolio.update_acc_buy(fill, acc, positions)
        self.assertEqual(local_portfolio.acc.position, fill.quantity)
        time.sleep(1)
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        acc = self.local_oa_execution_handler.get_account_summary()
        positions = self.local_oa_execution_handler.get_positions()

        local_portfolio.update_acc_sell(fill, acc, positions)
        self.assertEqual(local_portfolio.acc.position, 0.0)
        time.sleep(1)
        self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        acc = self.local_oa_execution_handler.get_account_summary()
        positions = self.local_oa_execution_handler.get_positions()

        local_portfolio.update_acc_sell(fill, acc, positions)
        self.assertEqual(local_portfolio.acc.position, -fill.quantity)
        time.sleep(1)
        self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        acc = self.local_oa_execution_handler.get_account_summary()
        positions = self.local_oa_execution_handler.get_positions()
        local_portfolio.update_acc_buy(fill, acc, positions)
        time.sleep(1)
        self.assertEqual(local_portfolio.acc.position, 0.0)

    def test_portfolio_update_limit_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_buy(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(1)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_buy(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNone(local_portfolio.limit_order)

    def test_update_positions_from_fill(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        init_acc = local_portfolio.acc
        # test for bad order
        filled_from_oanda = self.local_oa_execution_handler.create_order(self.symbol, 1000 * 1000 * 1000 * 1000 * 1000)
        self.assertIsNone(filled_from_oanda)
        local_portfolio.update_positions_from_fill(filled_from_oanda)
        self.assertEqual(local_portfolio.acc.position, init_acc.position)

        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        if not self.local_events.empty():
            fill = self.local_events.get()
            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, fill.quantity)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, 0.0)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, -fill.quantity)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, 0.0)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)

    def test_portfolio_update_limit_order_prices_buy(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.0001
        limit_price = local_portfolio.calculate_stop_loss(1.6534, limit_order_size, "BUY")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="BUY", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        print("limit order: ", local_portfolio.limit_order)
        new_entry_price = local_portfolio.calculate_stop_loss(1.6534, local_portfolio.limit_order.limit_order_size,
                                                              "BUY") - 0.00001  # new entry price is lower than limit price
        local_portfolio.update_limit_order_prices(new_entry_price)
        self.assertEqual(local_portfolio.limit_order.entry_price, entry_price)
        self.assertEqual(local_portfolio.limit_order.limit_price, limit_price)

        new_entry_price = entry_price + 0.00001  # new current price is higher than former entry price

        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, new_entry_price)
        self.assertNotEqual(local_portfolio.limit_order.limit_price, limit_price)

    def test_portfolio_update_limit_order_prices_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.0001
        limit_price = local_portfolio.calculate_stop_loss(1.6534, limit_order_size, "SELL")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="SELL", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        print("limit order: ", local_portfolio.limit_order)
        new_entry_price = local_portfolio.calculate_stop_loss(1.6534, local_portfolio.limit_order.limit_order_size,
                                                              "SELL") + 0.00001
        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, entry_price)
        self.assertEqual(local_portfolio.limit_order.limit_price, limit_price)

        new_entry_price = entry_price - 0.00001

        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, new_entry_price)
        self.assertNotEqual(local_portfolio.limit_order.limit_price, limit_price)

    def test_portfolio_check_limit_order_buy(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.001
        limit_price = local_portfolio.calculate_stop_loss(entry_price, limit_order_size, "BUY")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="BUY", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        new_entry_price = local_portfolio.calculate_stop_loss(entry_price, local_portfolio.limit_order.limit_order_size,
                                                              "BUY") - 0.00001  # new entry price is lower than limit price

        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)
        self.assertTrue(limit_triggered)
        self.assertEqual(order.direction, "EXIT")

        new_entry_price = entry_price - 0.00001  # new current price is lower than former entry price but higher than limit price
        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)

        self.assertFalse(limit_triggered)
        self.assertIsNone(order)

    def test_portfolio_check_limit_order_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.001
        limit_price = local_portfolio.calculate_stop_loss(entry_price, limit_order_size, "SELL")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="SELL", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        new_entry_price = local_portfolio.calculate_stop_loss(entry_price, local_portfolio.limit_order.limit_order_size,
                                                              "SELL") + 0.00001  # new entry price is higher than limit price

        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)

        self.assertTrue(limit_triggered)
        self.assertEqual(order.direction, "EXIT")

        new_entry_price = entry_price + 0.00001  # new current price is higher than former entry price but lower than
        # limit price
        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)
        self.assertFalse(limit_triggered)
        self.assertIsNone(order)


class TestPortfolio(unittest.TestCase):
    symbol = "AUD_USD"
    limit_order_size = 0.0001
    buy_order = OrderEvent(symbol, "MKT", 'BUY', date=dt.datetime.now(), limit_order_size=limit_order_size)
    sell_order = OrderEvent(symbol, "MKT", 'SELL', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)
    exit_order = OrderEvent(symbol, "MKT", 'EXIT', date=dt.datetime.now(),
                            limit_order_size=limit_order_size)

    def setUp(self) -> None:
        warnings.simplefilter("ignore", ResourceWarning)
        OandaExecutionHandler.close_positions()
        self.local_events = q.Queue()
        self.local_oa_execution_handler = OandaExecutionHandler(self.local_events)
        self.symbol = "AUD_USD"

    def test_portfolio_init_updates_positions(self):

        oanda_position = self.local_oa_execution_handler.get_positions()
        acc_summary = self.local_oa_execution_handler.get_account_summary()
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)

        self.assertEqual(local_portfolio.acc.position, 0.0)
        self.assertAlmostEqual(local_portfolio.acc.cash,
                               (float(acc_summary['balance']) *
                                local_portfolio.last_used_exchange_rate), delta=10)

        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        fill = self.local_events.get()
        local_portfolio.update_fill(fill)
        prev_acc = local_portfolio.acc
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.assertEqual(local_portfolio.acc.position, prev_acc.position)
        self.assertAlmostEqual(local_portfolio.acc.cash, prev_acc.cash, delta=10)

        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)
        fill = self.local_events.get()
        local_portfolio.update_fill(fill)

        prev_acc = local_portfolio.acc
        time.sleep(2)
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.assertEqual(local_portfolio.acc.position, prev_acc.position)
        self.assertAlmostEqual(local_portfolio.acc.cash, prev_acc.cash, delta=10)

    def test_portfolio_get_last_close_price(self):
        last_close_price = self.local_oa_execution_handler.get_last_close_price(self.symbol)
        self.assertIsNotNone(last_close_price)

    def test_portfolio_update_signal_returns_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, print(self.local_oa_execution_handler.get_last_close_price(self.symbol)))
        try:
            order = self.local_events.get()
        except q.Empty:
            order = None
        self.assertIsNotNone(order)

    def testt_portfolio_update_signal_into_position(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, print(self.local_oa_execution_handler.get_last_close_price(self.symbol)))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "BUY")

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, print(self.local_oa_execution_handler.get_last_close_price(self.symbol)))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "SELL")

    def test_portfolio_update_signal_out_of_position(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        self.assertIs(self.local_events.empty(), True)

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertIs(self.local_events.empty(), True)
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()
        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "EXIT")

        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        order = self.local_events.get()

        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "EXIT")

    def test_portfolio_update_acc_buy_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_buy(fill, acc, positions)
            self.assertEqual(local_portfolio.acc.position, fill.quantity)
            time.sleep(1)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            self.assertEqual(local_portfolio.acc.position, 0.0)
            time.sleep(1)
            self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            self.assertEqual(local_portfolio.acc.position, -fill.quantity)
            time.sleep(1)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()
            local_portfolio.update_acc_buy(fill, acc, positions)
            time.sleep(1)
            self.assertEqual(local_portfolio.acc.position, 0.0)

    def test_portfolio_update_limit_order(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_buy(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(1)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_sell(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_acc_buy(fill, acc, positions)
            local_portfolio.update_limit_order(positions, fill)
            self.assertIsNone(local_portfolio.limit_order)

    def test_update_positions_from_fill(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        init_acc = local_portfolio.acc
        # test for bad order
        filled_from_oanda = self.local_oa_execution_handler.create_order(self.symbol, 1000 * 1000 * 1000 * 1000 * 1000)
        self.assertIsNone(filled_from_oanda)
        local_portfolio.update_positions_from_fill(filled_from_oanda)
        self.assertEqual(local_portfolio.acc.position, init_acc.position)

        self.local_oa_execution_handler.execute_order(self.buy_order, local_portfolio)
        if not self.local_events.empty():
            fill = self.local_events.get()
            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, fill.quantity)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, 0.0)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.sell_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, -fill.quantity)
            self.assertIsNotNone(local_portfolio.limit_order)
            self.assertEqual(local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(local_portfolio.limit_order.direction, fill.direction)
            time.sleep(2)
            self.local_oa_execution_handler.execute_order(self.exit_order, local_portfolio)

        if not self.local_events.empty():
            fill = self.local_events.get()
            acc = self.local_oa_execution_handler.get_account_summary()
            positions = self.local_oa_execution_handler.get_positions()

            local_portfolio.update_positions_from_fill(fill)
            self.assertEqual(local_portfolio.acc.position, 0.0)
            self.assertIsNone(local_portfolio.limit_order)
            time.sleep(2)

    def test_portfolio_update_limit_order_prices_buy(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.0001
        limit_price = local_portfolio.calculate_stop_loss(1.6534, limit_order_size, "BUY")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="BUY", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        print("limit order: ", local_portfolio.limit_order)
        new_entry_price = local_portfolio.calculate_stop_loss(1.6534, local_portfolio.limit_order.limit_order_size,
                                                              "BUY") - 0.00001  # new entry price is lower than limit price
        local_portfolio.update_limit_order_prices(new_entry_price)
        self.assertEqual(local_portfolio.limit_order.entry_price, entry_price)
        self.assertEqual(local_portfolio.limit_order.limit_price, limit_price)

        new_entry_price = entry_price + 0.00001  # new current price is higher than former entry price

        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, new_entry_price)
        self.assertNotEqual(local_portfolio.limit_order.limit_price, limit_price)

    def test_portfolio_update_limit_order_prices_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.0001
        limit_price = local_portfolio.calculate_stop_loss(1.6534, limit_order_size, "SELL")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="SELL", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        print("limit order: ", local_portfolio.limit_order)
        new_entry_price = local_portfolio.calculate_stop_loss(1.6534, local_portfolio.limit_order.limit_order_size,
                                                              "SELL") + 0.00001
        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, entry_price)
        self.assertEqual(local_portfolio.limit_order.limit_price, limit_price)

        new_entry_price = entry_price - 0.00001

        local_portfolio.update_limit_order_prices(new_entry_price)

        self.assertEqual(local_portfolio.limit_order.entry_price, new_entry_price)
        self.assertNotEqual(local_portfolio.limit_order.limit_price, limit_price)

    def test_portfolio_check_limit_order_buy(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.001
        limit_price = local_portfolio.calculate_stop_loss(entry_price, limit_order_size, "BUY")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="BUY", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        new_entry_price = local_portfolio.calculate_stop_loss(entry_price, local_portfolio.limit_order.limit_order_size,
                                                              "BUY") - 0.00001  # new entry price is lower than limit price

        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)
        self.assertTrue(limit_triggered)
        self.assertEqual(order.direction, "EXIT")

        new_entry_price = entry_price - 0.00001  # new current price is lower than former entry price but higher than limit price
        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)

        self.assertFalse(limit_triggered)
        self.assertIsNone(order)

    def test_portfolio_check_limit_order_sell(self):
        local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        entry_price = 1.6534
        limit_order_size = 0.001
        limit_price = local_portfolio.calculate_stop_loss(entry_price, limit_order_size, "SELL")
        local_portfolio.acc = vbt.pf_enums.AccountState(
            cash=0.0,
            position=-1000.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=0.0
        )
        local_portfolio.limit_order = LimitOrderEvent(symbol='EUR_USD', order_type="LMT",
                                                      direction="SELL", limit_price=limit_price,
                                                      entry_price=entry_price, date=dt.datetime.date,
                                                      limit_order_size=limit_order_size)
        new_entry_price = local_portfolio.calculate_stop_loss(entry_price, local_portfolio.limit_order.limit_order_size,
                                                              "SELL") + 0.00001  # new entry price is higher than limit price

        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)

        self.assertTrue(limit_triggered)
        self.assertEqual(order.direction, "EXIT")

        new_entry_price = entry_price + 0.00001  # new current price is higher than former entry price but lower than
        # limit price
        limit_triggered, order = local_portfolio.check_limit_order(new_entry_price)
        self.assertFalse(limit_triggered)
        self.assertIsNone(order)


class TestPipelineFlowSim(unittest.TestCase):
    def setUp(self) -> None:
        OandaExecutionHandler.close_positions()
        self.symbol = currency
        self.local_events = q.Queue()
        self.local_oa_execution_handler = SimulatedExecutionHandler(self.local_events, data, init_cash=100000)
        self.local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        warnings.simplefilter("ignore", ResourceWarning)
        time.sleep(2)

    def test_buy_close(self):
        # open a long position

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

        # close the position
        time.sleep(2)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is closed
        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        # check that the limit order is cancelled
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_sell_close(self):
        # open a short position
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, -fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

        # close the position
        time.sleep(2)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is closed
        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        # check that the limit order is cancelled
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_buy_sell_buy_sell(self):
        # open a long position
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)
        for i in range(3):
            # open a short position
            time.sleep(2)
            signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                                 signal_type="SHORT",
                                 strength=1.0, limit_order_size=0.0001)
            self.local_portfolio.update_signal(signal,
                                               self.local_oa_execution_handler.get_last_close_price(self.symbol))

            self.assertFalse(self.local_events.empty())
            order = self.local_events.get()
            self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

            self.assertFalse(self.local_events.empty())
            fill = self.local_events.get()
            self.local_portfolio.update_fill(fill)

            # check that the position is open
            self.assertLess(self.local_portfolio.acc.position, 0.0)
            self.assertGreater(self.local_portfolio.acc.cash, 0.0)
            # check that the limit order is set
            self.assertIsNotNone(self.local_portfolio.limit_order)
            self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

            # open a long position
            time.sleep(2)
            signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                                 strength=1.0, limit_order_size=0.0001)
            self.local_portfolio.update_signal(signal,
                                               self.local_oa_execution_handler.get_last_close_price(self.symbol))

            self.assertFalse(self.local_events.empty())
            order = self.local_events.get()
            self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

            self.assertFalse(self.local_events.empty())
            fill = self.local_events.get()
            self.local_portfolio.update_fill(fill)

            # check that the position is open
            self.assertGreater(self.local_portfolio.acc.position, 0.0)
            self.assertLess(self.local_portfolio.acc.cash, 0.0)
            # check that the limit order is set
            self.assertIsNotNone(self.local_portfolio.limit_order)
            self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

    def test_check_limit_order_buy_no_conflict(self):
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "BUY") - 0.00001
        self.local_portfolio.update_signal(signal, sim_price)
        order = self.local_events.get()

        self.assertEqual(order.order_type, "LMT")
        self.assertEqual(order.direction, "EXIT")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_check_limit_order_sell_no_conflict(self):
        time.sleep(2)
        symbol = "AUD_USD"
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry,
                                                             self.local_portfolio.limit_order.limit_order_size,
                                                             "SELL") + 0.00001
        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "LMT")
        self.assertEqual(order.direction, "EXIT")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_check_limit_order_buy_conflict_with_sell(self):
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "BUY") - 0.00001

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "SELL")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertLess(self.local_portfolio.acc.position, 0.0)
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, "SELL")

    def test_check_limit_order_sell_conflict_with_buy(self):
        self.local_oa_execution_handler.get_current_price(self.symbol)

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "SELL") + 0.00001

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "BUY")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertGreater(self.local_portfolio.acc.position, 0.0)
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, "BUY")


class TestPipelineFlow(unittest.TestCase):
    def setUp(self) -> None:
        OandaExecutionHandler.close_positions()
        self.symbol = "AUD_USD"
        self.local_events = q.Queue()
        self.local_oa_execution_handler = OandaExecutionHandler(self.local_events)
        self.local_portfolio = Portfolio(self.local_events, self.local_oa_execution_handler)
        warnings.simplefilter("ignore", ResourceWarning)
        time.sleep(2)

    def test_buy_close(self):
        # open a long position

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))
        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

        # close the position
        time.sleep(2)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="LONG_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is closed
        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        # check that the limit order is cancelled
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_sell_close(self):
        # open a short position
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, -fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

        # close the position
        time.sleep(2)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                             signal_type="SHORT_EXIT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is closed
        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        # check that the limit order is cancelled
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_buy_sell_buy_sell(self):
        # open a long position
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        # check that the position is open
        self.assertEqual(self.local_portfolio.acc.position, fill.quantity)
        # check that the limit order is set
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)
        for i in range(3):
            # open a short position
            time.sleep(2)
            signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(),
                                 signal_type="SHORT",
                                 strength=1.0, limit_order_size=0.0001)
            self.local_portfolio.update_signal(signal,
                                               self.local_oa_execution_handler.get_last_close_price(self.symbol))

            self.assertFalse(self.local_events.empty())
            order = self.local_events.get()
            self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

            self.assertFalse(self.local_events.empty())
            fill = self.local_events.get()
            self.local_portfolio.update_fill(fill)

            # check that the position is open
            self.assertLess(self.local_portfolio.acc.position, 0.0)
            self.assertGreater(self.local_portfolio.acc.cash, 0.0)
            # check that the limit order is set
            self.assertIsNotNone(self.local_portfolio.limit_order)
            self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

            # open a long position
            time.sleep(2)
            signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                                 strength=1.0, limit_order_size=0.0001)
            self.local_portfolio.update_signal(signal,
                                               self.local_oa_execution_handler.get_last_close_price(self.symbol))

            self.assertFalse(self.local_events.empty())
            order = self.local_events.get()
            self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

            self.assertFalse(self.local_events.empty())
            fill = self.local_events.get()
            self.local_portfolio.update_fill(fill)

            # check that the position is open
            self.assertGreater(self.local_portfolio.acc.position, 0.0)
            self.assertLess(self.local_portfolio.acc.cash, 0.0)
            # check that the limit order is set
            self.assertIsNotNone(self.local_portfolio.limit_order)
            self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
            self.assertEqual(self.local_portfolio.limit_order.direction, fill.direction)

    def test_check_limit_order_buy_no_conflict(self):
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "BUY") - 0.00001
        self.local_portfolio.update_signal(signal, sim_price)
        order = self.local_events.get()

        self.assertEqual(order.order_type, "LMT")
        self.assertEqual(order.direction, "EXIT")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_check_limit_order_sell_no_conflict(self):
        time.sleep(2)
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry,
                                                             self.local_portfolio.limit_order.limit_order_size,
                                                             "SELL") + 0.00001
        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "LMT")
        self.assertEqual(order.direction, "EXIT")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertEqual(self.local_portfolio.acc.position, 0.0)
        self.assertIsNone(self.local_portfolio.limit_order)

    def test_check_limit_order_buy_conflict_with_sell(self):
        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "BUY") - 0.00001

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "SELL")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertLess(self.local_portfolio.acc.position, 0.0)
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, "SELL")

    def test_check_limit_order_sell_conflict_with_buy(self):
        self.local_oa_execution_handler.get_current_price(self.symbol)

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="SHORT",
                             strength=1.0, limit_order_size=0.0001)
        self.local_portfolio.update_limit_order_prices(self.local_oa_execution_handler.get_current_price(self.symbol))
        self.local_portfolio.update_signal(signal, self.local_oa_execution_handler.get_last_close_price(self.symbol))

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)

        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertIsNotNone(self.local_portfolio.limit_order)

        entry = self.local_portfolio.limit_order.entry_price
        sim_price = self.local_portfolio.calculate_stop_loss(entry, self.local_portfolio.limit_order.limit_order_size,
                                                             "SELL") + 0.00001

        signal = SignalEvent(strategy_id="TSM1", symbol=self.symbol, datetime=datetime.utcnow(), signal_type="LONG",
                             strength=1.0, limit_order_size=0.0001)

        self.local_portfolio.update_signal(signal, sim_price)

        self.assertFalse(self.local_events.empty())
        order = self.local_events.get()

        self.assertEqual(order.order_type, "MKT")
        self.assertEqual(order.direction, "BUY")

        time.sleep(2)
        self.local_oa_execution_handler.execute_order(order, self.local_portfolio)
        self.assertFalse(self.local_events.empty())
        fill = self.local_events.get()
        self.local_portfolio.update_fill(fill)

        self.assertGreater(self.local_portfolio.acc.position, 0.0)
        self.assertIsNotNone(self.local_portfolio.limit_order)
        self.assertEqual(self.local_portfolio.limit_order.order_type, "LMT")
        self.assertEqual(self.local_portfolio.limit_order.direction, "BUY")


if __name__ == "__main__":
    unittest.main()
