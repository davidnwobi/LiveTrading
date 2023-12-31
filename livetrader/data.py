from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .event import MarketEvent
from .oadata import OAData


class DataHandler(object, metaclass=ABCMeta):
    """
    DataHandler is an abstract base class providing an interface for all subsequent (inherited)
    data handlers (both live and historic).
    The goal of a derived DataHandler object is to output a generated set of bars (OHLCVI)
    for each symbol requested. This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus, a historic and live system will be treated
    identically by the rest of the backtesting suite.
    """

    @staticmethod
    def remove_non_trading_days_forex(data):
        general_mask = (data.index.weekday >= 0) & (data.index.weekday <= 3)
        friday_mask = (data.index.weekday == 4) & (data.index.hour <= 21)
        sunday_mask = (data.index.weekday == 6) & (data.index.hour > 21)
        return data[general_mask | friday_mask | sunday_mask]

    @staticmethod
    def resample_dataframe(df, rule):
        resampled_open = df['Open'].resample(rule).first()
        resampled_high = df['High'].resample(rule).max()
        resampled_low = df['Low'].resample(rule).min()
        resampled_close = df['Close'].resample(rule).last()
        resampled_volume = df['Volume'].resample(rule).mean()

        resampled_df = pd.DataFrame({
            'Open': resampled_open,
            'High': resampled_high,
            'Low': resampled_low,
            'Close': resampled_close,
            'Volume': resampled_volume
        })

        return resampled_df

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume, or OI from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol in a tuple OHLCVI format:
        (datetime, open, high, low, close, volume, open interest)
        """
        raise NotImplementedError("Should implement update_bars()")


class ForexHistoricHDFDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live trading interface.
    """

    def __init__(self, events, data_dir: Path, symbol_list: List[str], freq):
        """
        Initializes the historic data handler by requesting the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form 'symbol.csv', where symbol is a string in the list.

        Parameters:
        - events: The Event Queue.
        - csv_dir: Absolute directory path to the CSV files.
        - symbol_list: A list of symbol strings.
        """
        self.events = events
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.symbol_data = dict()
        self.latest_symbol_data = dict()
        self.returns = dict()
        self.continue_backtest = True
        self.freq = freq
        self.start_date = None
        self._open_convert_hdf_files()

    def _open_convert_hdf_files(self):
        """
        Opens the CSV files from the data directory, converting them into pandas DataFrames within a symbol dictionary.
        For this handler, The data is taken from Oanda
        """

        for symbol in self.symbol_list:
            # Load the HDF file with no header information, indexed on date
            ask_price = OAData.from_hdf(self.data_dir / symbol / (symbol + "_" + "A.hdf")).get()
            ask_price = self.remove_non_trading_days_forex(ask_price)
            ask_price = self.resample_dataframe(ask_price, self.freq)
            ask_price["returns"] = pd.Series(np.log(ask_price.Close / ask_price.Close.shift(1)), name="returns")
            self.returns[symbol] = pd.Series(np.log(ask_price.Close / ask_price.Close.shift(1)), name="returns")
            self.start_date = ask_price.index[0]
            self.symbol_data[symbol] = ask_price.iterrows()
            self.latest_symbol_data[symbol] = []

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed.
        """
        for bar in self.symbol_data[symbol]:
            yield bar

    def get_latest_bar(self, symbol):
        """
        Returns the last bar from the latest_symbol_data list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol_data list, or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Volume, or OI values from the Pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol, val_type, N=1):
        """
        Returns the last N bar values from the latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure for all symbols in the symbol list.
        """
        for symbol in self.symbol_list:
            try:
                bar = next(self._get_new_bar(symbol))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[symbol].append(bar)
            self.events.put(MarketEvent())
