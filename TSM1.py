from statsmodels.tsa.ar_model import AutoReg
from scipy import stats as st
import time
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from livetrader.event import SignalEvent, MarketEvent
from typing import Tuple
from queue import Queue
from livetrader.logger_config import flow_logger


def conf_interval(data: np.ndarray, t_crit: float = 1.96) -> np.ndarray:
    """
    Calculate the confidence interval for a given dataset.

    Parameters:
        data (np.ndarray): Input data.
        t_crit (float): Critical value for the t-distribution. Defaults to 1.96 (95% confidence interval).

    Returns:
        np.ndarray: Confidence interval limits.
    """
    limits = np.full(2, np.nan)
    mean = np.mean(data)
    std = np.std(data)
    interval = std / np.sqrt(len(data)) * t_crit
    limits[0] = mean + interval
    limits[1] = mean - interval
    return limits


def remove_non_trading_days_forex(data: pd.Series) -> pd.Series:
    """
    Remove non-trading days (weekends) from a Forex price series.

    Parameters:
        data (pd.Series): Price series.

    Returns:
        pd.Series: Price series with non-trading days removed.
    """
    general_mask = (data.index.weekday >= 0) & (data.index.weekday <= 3)
    friday_mask = (data.index.weekday == 4) & (data.index.hour <= 21)
    sunday_mask = (data.index.weekday == 6) & (data.index.hour > 21)
    return data[general_mask | friday_mask | sunday_mask]


class TradingStratsMark1:
    """
    Class representing a trading strategy that combines an autoregressive model with an Aroon indicator.

    The TradingStratsMark1 class utilizes an autoregressive model to predict future prices based on historical data.
    It takes into account the specified number of lags (comb_lag) and the length of history (comb_length) used for
    training the model. The predicted_points attribute stores an array of predicted points.

    In addition to the autoregressive model, this trading strategy incorporates the Aroon indicator.
    The Aroon indicator measures the strength and direction of a trend, providing insights into potential entry and
    exit points. The aroontimeperiod parameter determines the number of periods considered in the Aroon indicator.

    This class assumes that the input data is provided as a pandas DataFrame containing High, Low and Close prices.
    The frequency of the data is assumed to be 4 hours. The High and Low Price are down-sampled to 1 day and used to
    calculate the Aroon indicator. This is then up-sampled and aligned with the Close Price at 4h. When the
    Autoregressive model predicts a price above the upper confidence interval of 99.9%, the Aroon indicator is used
    to determine whether to enter a long position. When the Autoregressive model predicts a price below the lower
    confidence interval of 99.9%, the Aroon indicator is used to determine whether to enter a short position. The
    Aroon indicator is also used to determine whether to exit a position.

    Attributes:
        data (pd.DataFrame): Input data. Need to be at least 201 rows longer that comb_length.
        comb_lag (int): Number of lags to consider in the AutoReg model.
        comb_length (int): Length of history used for training the model.
        predicted_points (np.ndarray): Array of predicted points.
        aroontimeperiod (int): Number of periods to consider in the Aroon indicator.
    """

    def __init__(self, events: Queue, symbol: str, data: pd.DataFrame, comb_lag: int, comb_length: int,
                 aroontimeperiod: int = 14):
        """
        Initialize the TradingStratsMark1 object.

        Parameters:
            events (Queue): Event queue.
            symbol (str): Symbol of the asset.
            data (pd.DataFrame): Input data.
            comb_lag (int): Number of lags to consider in the AutoReg model.
            comb_length (int): Length of history used for training the model.
            aroontimeperiod (int): Number of periods to consider in Aroon indicator. Defaults to 14.
        """
        self.events = events
        self.symbol = symbol
        self.data = data
        self.comb_lag = comb_lag
        self.comb_length = comb_length
        self.predicted_points = self.init_predicted_points()
        self.debug_dict = {}
        self.position = "None"

        self.aroontimeperiod = aroontimeperiod

    def __eq__(self, other):
        return self.__dict__ == other

    @staticmethod
    def forecast_ar_1_test_parallel(history: pd.Series, lags: int) -> float:
        """
        Forecast the next value using an ARIMA model for a given index.

        Parameters:
            history (pd.Series): Historical data concatenated with test data for model training and forecasting.
            lags (int): Number of lags to consider in the AutoReg model.

        Returns:
            float: The forecasted next value.
        """
        model = AutoReg(history, lags).fit()
        next_pred = model.forecast(steps=1)
        return next_pred[0]

    def init_predicted_points(self, num_init_points: int = 200) -> np.ndarray:
        """
        Initialize the array of predicted points.
        Remember to generate a predicted point you need to have at least comb_length + 1 points in the input data.
        So to generate num_init_points predicted points, you need to have num_init_points + comb_length + 1 points

        Parameters:
            num_init_points (int): Number of initial points to generate. Defaults to 200.

        Returns:
            np.ndarray: Array of predicted points.
        """
        data = np.log(self.data.Close).diff().dropna()

        length_for_training = self.comb_length
        y_pred_test_wfv = np.full(num_init_points, np.nan)
        len_pred = len(y_pred_test_wfv)

        for i in range(1,len_pred):
            y_pred_test_wfv[i] = self.forecast_ar_1_test_parallel(
                data[-length_for_training + i - len_pred:i - len_pred].values, self.comb_lag)
        return y_pred_test_wfv

    def calc_aroon(self) -> Tuple[float, float]:
        """
        Calculate the Aroon indicator values.

        Returns:
            Tuple[float, float]: Aroonup and Aroondown values.
        """
        buffer = 100
        high = remove_non_trading_days_forex(self.data.High[-buffer:].resample('4H').max())
        low = remove_non_trading_days_forex(self.data.Low[-buffer:].resample('4H').min())
        aroon = vbt.indicator('talib:AROON').run(high, low)
        aroonup_upsampled = aroon.aroonup.vbt.resample_closing("H").ffill()  # Changing to H for testing
        aroondown_upsampled = aroon.aroondown.vbt.resample_closing("H").ffill()  #
        aroonup_upsampled.index = aroonup_upsampled.index
        aroonup_upsampled = aroonup_upsampled.reindex_like(self.data.Close).ffill()
        aroondown_upsampled.index = aroondown_upsampled.index
        aroondown_upsampled = aroondown_upsampled.reindex_like(self.data.Close).ffill()
        return aroonup_upsampled.values[-1], aroondown_upsampled.values[-1]

    def calc_next_pred(self) -> float:
        """
        Calculate the next predicted value.

        Returns:
            float: The next predicted value.
        """
        data = np.log(self.data.Close).diff()
        length_for_training = self.comb_length
        y_pred_test_wfv = np.full(1, np.nan)
        len_pred = len(y_pred_test_wfv)
        y_pred_test_wfv[0] = self.forecast_ar_1_test_parallel(data[-length_for_training - len_pred:-len_pred].values,
                                                              self.comb_lag)
        return y_pred_test_wfv[0]

    def calc_confidence_interval(self) -> np.ndarray:
        """
        Calculate the confidence interval for the predicted points.

        Returns:
            np.ndarray: Confidence interval limits.
        """
        t_crit = st.t.ppf(q=(1 - 0.999) / 2, df=self.comb_lag - 1)
        conf_int = conf_interval(self.predicted_points[:self.comb_lag], t_crit=t_crit)
        self.debug_dict['t_crit'] = t_crit
        self.debug_dict['conf_int'] = conf_int
        return conf_int

    def update_predictions(self) -> np.ndarray:
        """
        Update the array of predicted points by shifting and adding the next predicted value.

        Returns:
            np.ndarray: Updated array of predicted points.
        """
        self.predicted_points = np.roll(self.predicted_points, 1)
        self.predicted_points[0] = self.calc_next_pred()
        return self.predicted_points

    def signal_generator(self, prediction: float, aroonup: float, aroondown: float, intervals: np.ndarray,
                         position: str) -> SignalEvent:
        """
        Generate a signal event based on the prediction, Aroon indicator values, confidence intervals, and current position.

        Parameters:
            prediction (float): The next predicted value.
            aroonup (float): Aroonup value.
            aroondown (float): Aroondown value.
            intervals (np.ndarray): Confidence interval limits.
            position (str): Current position ('LONG' or 'SHORT').

        Returns:
            SignalEvent: Generated signal event.
        """
        buy_condition_aroon = (aroonup > 70) and (aroonup > aroondown) and (aroondown < 50)
        sell_condition_aroon = (aroondown > 70) and (aroondown > aroonup) and (aroonup < 50)
        if (prediction > intervals[1]) and buy_condition_aroon:
            return SignalEvent('TSM1', self.symbol, self.data.index[-1], "LONG", 1)
        elif (prediction < intervals[0]) and sell_condition_aroon:
            return SignalEvent('TSM1', self.symbol, self.data.index[-1], "SHORT", 1)
        elif not buy_condition_aroon and position == 'LONG':
            return SignalEvent('TSM1', self.symbol, self.data.index[-1], "LONG_EXIT", 0)
        elif not sell_condition_aroon and position == 'SHORT':
            return SignalEvent('TSM1', self.symbol, self.data.index[-1], "SHORT_EXIT", 0)
        else:
            return SignalEvent('TSM1', self.symbol, self.data.index[-1], "None", 0)

    def generate_signal(self, event: MarketEvent):
        """
        Generate a signal event based on the market event and current position. Puts the signal event into the event queue.

        Parameters:
            event (MarketEvent): Market event.
            position (str): Current position ('LONG' or 'SHORT').

        Returns:
            None
        """
        if event.type == 'MARKET':

            aroonup, aroondown = self.calc_aroon()

            self.debug_dict['aroonup'] = aroonup
            self.debug_dict['aroondown'] = aroondown

            next_pred = self.calc_next_pred()

            self.debug_dict['next_pred'] = next_pred

            signal = self.signal_generator(next_pred, aroonup, aroondown, self.calc_confidence_interval(), self.position)
            if signal.signal_type == "LONG" or signal.signal_type == "SHORT":
                self.position = signal.signal_type
                self.debug_dict["position"] = self.position
            else:
                self.position = "NONE"
                self.debug_dict["position"] = self.position

            signal.debug_dict = self.debug_dict
            self.debug_dict = {}

            self.update_predictions()
            self.events.put(signal)
            flow_logger.info(f"Signal generated: {signal}, Current position: {self.position}")
