import numpy as np
import pandas as pd
import vectorbtpro as vbt

def mask_function(data):
    date_range = data.index
    mask = ((date_range.weekday == 4) & (date_range.hour <=22)) | ((date_range.weekday == 5) & (date_range.hour <0)) | ((date_range.weekday == 6) & (date_range.hour >=22))
    return data[mask]

def time_breaks(df):
   # build complete timepline from start date to end date
    all_dates = pd.date_range(start=df.index[0],end=df.index[-1])

    # retrieve the dates that ARE in the original datset
    date_original = [dates.strftime("%Y-%m-%d %H:%M:%S") for dates in pd.to_datetime(df.index)]
    
    # retrieve the dates that ARE not in the original datset but are in the full timelien
    date_breaks = [dates for dates in all_dates.strftime("%Y-%m-%d %H:%M:%S").tolist() if not dates in date_original]
    return date_breaks

def time_breaks_h1(df, freq='H'):
   # build complete timepline from start date to end date
    all_dates = pd.date_range(start=df.index[0],end=df.index[-1], freq=freq)
    # retrieve the dates that ARE in the original datset
    date_original = df.index.copy()
    
    # retrieve the dates that ARE not in the original datset but are in the full timelien
    date_breaks = [dates.tz_convert(None) for dates in all_dates.tolist() if not dates in date_original and not (dates.weekday() == 6) and not((dates.weekday() == 5) and (dates.hour > 22))]
    
    return date_breaks

class OAData(vbt.Data):
    @classmethod
    def fetch_symbol(cls, symbol, **kwargs):
        """
        Fetches historical data for a given symbol using the `get_oanda_symbol` method of the OAData class.

        Args:
            symbol (str): The symbol to fetch data for.
            **kwargs: Additional keyword arguments to pass to the `get_oanda_symbol` method.

        Returns:
            tuple: A tuple containing the historical data for the specified symbol and additional information.

        """        
        returned_kwargs = dict(timestamp=pd.Timestamp.now())
        return OAData.get_oanda_symbol(symbol, **kwargs), returned_kwargs

    def update_symbol(self, symbol, **kwargs):
        """
        Updates the historical data for a given symbol by fetching new data based on the last available index.

        Args:
            symbol (str): The symbol to update data for.
            **kwargs: Additional keyword arguments to pass to the `fetch_symbol` method.

        Returns:
            tuple: A tuple containing the updated historical data for the specified symbol and additional information.

        """
        defaults = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        defaults["start"] = self.last_index[symbol]
        kwargs = vbt.merge_dicts(defaults, kwargs)
        return self.fetch_symbol(symbol, **kwargs)

    @staticmethod
    def get_oanda_symbol(symbol, period="H1", price="B", start=None, end=None, **kwargs):
        """
        Retrieves historical data for a given symbol from the OANDA API.

        Args:
            symbol (str): The symbol to retrieve data for.
            period (str, optional): The time period granularity of the data (default: "H1").
            price (str, optional): The type of price data to retrieve (default: "B").
            start (pd.Timestamp, optional): The start date of the data range (default: None).
            end (pd.Timestamp, optional): The end date of the data range (default: None).
            **kwargs: Additional keyword arguments to pass to the underlying OANDA API.

        Returns:
            pandas.DataFrame: The historical data for the specified symbol.

        Raises:
            ValueError: If less than three out of start, end, period, or freq are provided.

        """
        import tpqoa

        oanda = tpqoa.tpqoa("../oanda.cfg")
        if isinstance(start, pd.Timestamp):
            start = start.to_pydatetime().date().isoformat()
        if isinstance(end, pd.Timestamp):
            end = end.to_pydatetime().date().isoformat()
        print(start, end, symbol)
        data = oanda.get_history(
            instrument=symbol,
            start=start,
            end=end,
            granularity=period,
            price=price,
            localize=False,
        )
        data.index.name = "Datetime"
        data.index = data.index.tz_convert(None)
        data.rename(
            columns=dict(o="Open", h="High", c="Close", l="Low", volume="Volume"),
            inplace=True,
        )
        data = data.drop("complete", axis=1)
        return data
