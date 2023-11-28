import pandas as pd
import tpqoa
import numpy as np
from datetime import datetime
from .execution import ExecutionHandler
from .event import FillEvent, OrderEvent
from queue import Queue
from . import portfolio
import vectorbtpro as vbt
import typing as tp
import oandapyV20
import oandapyV20.endpoints.orders as orders
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
import oandapyV20.endpoints.positions as positions
import time
import configparser
from .logger_config import flow_logger, order_logger
import datetime as dt

# Load API credentials from configuration file

oanda_cfg_location = '../oanda.cfg'
class OandaExecutionHandler(ExecutionHandler):
    """
    Handles order execution via the Oanda v20 API.
    """

    def __init__(self, events: Queue, symbol='EUR_USD'):
        """
        Initialises the Oanda execution handler, setting the account
        ID and access token, as well as the API endpoint.
        """
        self.oanda = tpqoa.tpqoa(oanda_cfg_location)
        self.events = events

    @staticmethod
    def get_last_close_price(currency_pair):
        time_now = dt.datetime.now()
        time_hour = time_now.strftime("%H")
        if time_now.minute < 46:
            time_hour = (time_now - dt.timedelta(hours=1)).strftime("%H")
        try:
            data = pd.read_csv(f"{currency_pair}/{currency_pair}_{time_hour}.csv",
                               index_col="time", parse_dates=["time"]).iloc[-1].c
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File {currency_pair}_{time_hour}.csv not found")

    def get_account_summary(self):
        """
        Get account summary
        """
        return self.oanda.get_account_summary()

    def get_positions(self):
        """
        Get open positions
        """
        return self.oanda.get_positions()

    @staticmethod
    def create_order(instrument, units, max_retries=5, retry_delay=2, **kwargs):
        """
        Create an order with retry functionality.

        Parameters:
        - instrument: Instrument to open the order for
        - units: Number of units to buy or sell
        - max_retries: Maximum number of retries in case of connection issues (default: 5)
        - retry_delay: Delay in seconds between retries (default: 2)

        Returns:
        - Order response if successful, otherwise None
        """
        config = configparser.ConfigParser()
        config.read(oanda_cfg_location)
        account_id = config['oanda']['account_id']
        api_key = config['oanda']['access_token']

        order_params = {
            "order": {
                "units": units,
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }

        with oandapyV20.API(access_token=api_key) as api:
            for retry_count in range(max_retries + 1):
                try:
                    order_request = orders.OrderCreate(accountID=account_id, data=order_params)
                    api.request(order_request)
                    response = order_request.response

                    if response is not None and 'orderFillTransaction' in response:
                        # Order successfully created
                        order_logger.info(f"Order successfully created: {response['orderFillTransaction']}")
                        return response["orderFillTransaction"]

                except oandapyV20.exceptions.V20Error as e:
                    # Handle any potential OANDA API errors here
                    order_logger.error(f"OANDA API Error: {e}")
                    print(f"OANDA API Error: {e}")
                    return None
                except Exception as e:
                    # Handle other unexpected errors here
                    print(f"Unexpected Error: {e}")

                # Wait before retrying
                if retry_count < max_retries:
                    time.sleep(retry_delay)

            print("Failed to create order after maximum retries.")
            return None

    @staticmethod
    def get_current_price(currency_pair, price_type="ASK"):
        # Replace with your OANDA API credentials

        config = configparser.ConfigParser()
        config.read(oanda_cfg_location)
        account_id = config['oanda']['account_id']
        api_key = config['oanda']['access_token']

        # Create a PricingStream object
        params = {
            "instruments": currency_pair,  # Replace with the desired instrument(s)
            "accountId": account_id
        }
        r = PricingStream(accountID=account_id, params=params)

        # Create an empty DataFrame to store the ticks
        df_ticks = dict()
        current_hour = datetime.now()

        # Start the pricing stream
        try:
            with oandapyV20.API(access_token=api_key) as api:
                for ticks in api.request(r):
                    if ticks["type"] == "PRICE":
                        if price_type == "ASK":
                            return float(ticks["asks"][0]["price"])
                        elif price_type == "BID":
                            return float(ticks["bids"][0]["price"])
                        else:
                            raise V20Error(400, "price_type must be 'ASK' or 'BID'")

        except V20Error as e:
            print("Error: {}".format(e))

    def get_current_time(self):
        return datetime.now()

    @staticmethod
    def close_positions():
        """
        Close all open positions

        :return:
        """

        # Load API credentials from configuration file
        config = configparser.ConfigParser()
        config.read(oanda_cfg_location)
        account_id = config['oanda']['account_id']
        api_key = config['oanda']['access_token']

        try:
            # Get a list of all open positions
            with oandapyV20.API(access_token=api_key) as api:
                position_list_request = positions.OpenPositions(accountID=account_id)
                api.request(position_list_request)
                response = position_list_request.response
                positions_list = response.get('positions', [])
                if len(positions_list) == 0:
                    print("No open positions to close.")
                else:
                    for position in positions_list:
                        instrument = position['instrument']
                        units = position['long']['units'] if float(position['long']['units']) != 0 else \
                            position['short'][
                                'units']

                        # Close the position
                        data = {'longUnits': 'ALL'} if float(position['long']['units']) != 0 else {'shortUnits': 'ALL'}

                        request = positions.PositionClose(accountID=account_id, instrument=instrument, data=data)
                        response = api.request(request)
                        order_logger.info(f"Closed position for {instrument}. Response: {response}")
                        if 'errorMessage' in response:
                            print(f"Failed to close position for {instrument}. Error: {response['errorMessage']}")
                        else:
                            print(f"Closed position for {instrument}. Units: {units}")

        except V20Error as e:
            print("Error: {}".format(e))

    @staticmethod
    def create_fill_event(filled, limit_order_size):
        if float(filled['units']) > 0:
            fill_event = FillEvent(timeindex=datetime.fromisoformat(filled['time'][:-4]),
                                   symbol=filled['instrument'], exchange='Oanda', quantity=float(filled['units']),
                                   direction="BUY", fill_cost=float(filled["price"]),
                                   commission=float(filled["halfSpreadCost"]),
                                   usd_exchange_rate=OandaExecutionHandler.get_current_price("GBP_USD"),
                                   limit_order_size=limit_order_size)
        else:

            fill_event = FillEvent(timeindex=datetime.fromisoformat(filled['time'][:-4]),
                                   symbol=filled['instrument'], exchange='Oanda',
                                   quantity=abs(float(filled['units'])),
                                   direction="SELL", fill_cost=float(filled["price"]),
                                   commission=float(filled["halfSpreadCost"]),
                                   usd_exchange_rate=OandaExecutionHandler.get_current_price("GBP_USD"),
                                   limit_order_size=limit_order_size)
        return fill_event

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

    def execute_order(self, event: OrderEvent, pf: tp.Optional[portfolio.Portfolio] = None) -> None:
        """
        Creates the order on Oanda using the tpqoa API.

        Args:
            event: Contains an Event object with order information.
            pf: The Portfolio object.
        Returns:
            None
        """
        filled = None
        current_price = OandaExecutionHandler.get_current_price(event.symbol)
        pf.update_acc_exchange()
        if event.type == 'ORDER':  # Needs to be changed to work on the next close
            order_size = self.parse_order(event.direction, pf, current_price)
            if not np.isnan(order_size):
                filled = OandaExecutionHandler.create_order(event.symbol, units=int(order_size))
        if filled is not None:
            flow_logger.info(f"Order executed and filled: {filled}")
            fill_event = OandaExecutionHandler.create_fill_event(filled, event.limit_order_size)
            self.events.put(fill_event)
        else:
            flow_logger.warning(f"Order not executed: {event}")
            self.events.put(None)
