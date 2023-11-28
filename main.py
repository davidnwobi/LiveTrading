from livetrader.event import MarketEvent, SignalEvent, FillEvent
import queue as q
from livetrader.portfolio import Portfolio
from TSM1 import TradingStratsMark1
import vectorbtpro as vbt
import pandas as pd
from pathlib import Path
from sim_execution import SimulatedExecutionHandler
import numpy as np
from tqdm import tqdm

# Define constants
DATA_PATH = 'D:/Data'
CURRENCY = 'SPX500_USD'
INITIAL_DATA_POINTS = 403

# Load data
data = vbt.HDFData.fetch(Path(DATA_PATH) / CURRENCY / (CURRENCY + "_" + "A.hdf") / CURRENCY).get()


# Data generator function
def data_generator(data, len_data):
    for i in range(len(data) - len_data):
        yield data.iloc[i:i + len_data]


# Create price area
def make_price_area(close_price):
    return vbt.pf_enums.PriceArea(open=close_price, high=close_price, low=close_price, close=close_price)


# Create execution state
def make_exec_state(account_state, price_area):
    return vbt.pf_enums.ExecState(
        cash=account_state.cash,
        position=0.0,
        debt=account_state.debt,
        locked_cash=account_state.locked_cash,
        free_cash=account_state.free_cash,
        val_price=price_area.close,
        value=account_state.cash
    )


# Event queue
events = q.Queue()

# Setting up the Strategy
data_gen = data_generator(data, INITIAL_DATA_POINTS)
current_data = next(data_gen)
TSM1 = TradingStratsMark1(events, CURRENCY, current_data, 32, 188)

# Setting up VectorbtPro to track and record orders
order_records = np.empty((len(data), 1), dtype=vbt.pf_enums.order_dt)
order_counts = np.full(1, 0, dtype=np.int_)
order_counter = 1
sim_execution_handler = SimulatedExecutionHandler(events, data, 100000)
portfolio = Portfolio(events, sim_execution_handler)

# Main loop testing on simulated data
for _ in tqdm(range(0, len(data) - INITIAL_DATA_POINTS - 63000), desc="Progress: "):
    try:
        data_updated = False
        # Notify strategy of new data
        TSM1.generate_signal(MarketEvent())

        if not events.empty():
            # Update portfolio
            signal = events.get()
            portfolio.update_limit_order_prices(sim_execution_handler.get_current_price(CURRENCY))
            portfolio.update_signal(signal, sim_execution_handler.get_current_price(CURRENCY))

            if not events.empty():
                # Update data
                current_data = next(data_gen)
                data_updated = True
                TSM1.data = current_data
                sim_execution_handler.data = current_data

                # Execute order on new data
                order = events.get()
                sim_execution_handler.execute_order(order, portfolio)

                if not events.empty():
                    fill: FillEvent = events.get()
                    portfolio.update_fill(fill)

                    if fill:
                        execute_order_items = fill.execute_order_items
                        price_area = make_price_area(execute_order_items["price"])
                        exec_state = make_exec_state(execute_order_items["prev_acc_state"], price_area)

                        # Update order records
                        _, exec_state = vbt.pf_nb.process_order_nb(
                            0, 0, order_counter,
                            exec_state=exec_state,
                            order=vbt.pf_nb.order_nb(size=-1, size_type=vbt.pf_enums.SizeType.TargetPercent,
                                                     fees=0.00005 * portfolio.leverage, leverage=portfolio.leverage,
                                                     leverage_mode=vbt.pf_enums.LeverageMode.Eager, log=True),
                            order_records=order_records,
                            order_counts=order_counts,
                            price_area=price_area,
                            update_value=True
                        )
                        order_counter += 1

                # Price has moved, update limit order prices
                portfolio.update_limit_order_prices(sim_execution_handler.get_current_price(CURRENCY))
                # Check if limit order has been filled
                signal = SignalEvent('TSM1', CURRENCY, current_data.index[-1], "None", 0)
                portfolio.update_signal(signal, sim_execution_handler.get_current_price(CURRENCY))

                # Handle it if it has been filled
                if not events.empty():

                    order = events.get()
                    sim_execution_handler.execute_order(order, portfolio)

                    if not events.empty():
                        fill: FillEvent = events.get()
                        portfolio.update_fill(fill)

                        if fill:
                            execute_order_items = fill.execute_order_items
                            price_area = make_price_area(execute_order_items["price"])
                            exec_state = make_exec_state(execute_order_items["prev_acc_state"], price_area)

                            _, exec_state = vbt.pf_nb.process_order_nb(
                                0, 0, order_counter,
                                exec_state=exec_state,
                                order=vbt.pf_nb.order_nb(size=-1, size_type=vbt.pf_enums.SizeType.TargetPercent,
                                                         fees=0.00005 * portfolio.leverage,
                                                         leverage=portfolio.leverage,
                                                         leverage_mode=vbt.pf_enums.LeverageMode.Eager, log=True),
                                order_records=order_records,
                                order_counts=order_counts,
                                price_area=price_area,
                                update_value=True
                            )
                            order_counter += 1

            if not data_updated:
                current_data = next(data_gen)
                TSM1.data = current_data
                sim_execution_handler.data = current_data
                # Price has moved, update limit order prices
                signal = SignalEvent('TSM1', CURRENCY, current_data.index[-1], "None", 0)
                portfolio.update_limit_order_prices(sim_execution_handler.get_current_price(CURRENCY))
                portfolio.update_signal(signal, sim_execution_handler.get_current_price(CURRENCY))

                if not events.empty():
                    order = events.get()
                    sim_execution_handler.execute_order(order, portfolio)

                    if not events.empty():
                        fill: FillEvent = events.get()
                        portfolio.update_fill(fill)

                        if fill:
                            execute_order_items = fill.execute_order_items
                            price_area = make_price_area(execute_order_items["price"])
                            exec_state = make_exec_state(execute_order_items["prev_acc_state"], price_area)

                            _, exec_state = vbt.pf_nb.process_order_nb(
                                0, 0, order_counter,
                                exec_state=exec_state,
                                order=vbt.pf_nb.order_nb(size=-1, size_type=vbt.pf_enums.SizeType.TargetPercent,
                                                         fees=0.00005 * portfolio.leverage,
                                                         leverage=portfolio.leverage,
                                                         leverage_mode=vbt.pf_enums.LeverageMode.Eager, log=True),
                                order_records=order_records,
                                order_counts=order_counts,
                                price_area=price_area,
                                update_value=True
                            )
                            order_counter += 1

    except StopIteration:
        break

# Concatenate order records and save to CSV
df = pd.concat([pd.DataFrame(record.view(np.recarray)) for record in order_records])
df.to_csv("order_records.csv")

print(df)
