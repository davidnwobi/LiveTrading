import pandas as pd
import vectorbtpro as vbt
import numpy as np


class ModelData:


    def __init__(self, normal_data: pd.Series = None, model_data: pd.Series = None,  normal_index: pd.Series = None,
                 freq: str = 'H', forex: bool = True, resample=False) -> None:
        if not resample:
            if normal_data is None:
                self.model_returns = model_data
                self.normal_index = normal_index
                self.check_data(self.model_returns)
                self.normal_returns = self.model_to_normal(self.model_returns, self.normal_index)
            else:
                self.normal_data = normal_data
                self.normal_data.name = 'Close'
                self.model_data, self.normal_index = self.normal_to_model(self.normal_data, freq)
                self.normal_returns = self.log_returns(self.normal_data)
                self.model_returns, self.normal_returns_index = self.normal_to_model(self.normal_returns, freq=freq)
        else:
            if normal_data is None:
                raise ValueError("Model data can not be resampled")
            else:
                self.normal_data = normal_data.resample(freq).last()
                self.normal_data = normal_data
                self.normal_data.name = 'Close'
                self.model_data, self.normal_index = self.normal_to_model(self.normal_data, freq)
                self.normal_returns = self.log_returns(self.normal_data)
                self.model_returns, self.normal_returns_index = self.normal_to_model(self.normal_returns, freq=freq)

    def check_data(self, data):
        if data is None:
            raise ValueError("Data cannot be None. Expected a Pandas Series object.")
        
        if not isinstance(data, pd.Series):
            raise ValueError("Invalid data type. Expected a Pandas Series object.")
        
    def normal_to_model(self, normal_data: pd.Series, freq:str = 'H') -> pd.Series:
        """
        Convert non-continuous time data into returns in a format compatible with machine learning models.
        
        Args:
            close_price (pd.Series): Series containing close price data.
        
        Returns:
            pd.Series: Series containing model-compatible returns data.
        """
        from datetime import datetime, timedelta
        normal_data = normal_data
        normal_index  = normal_data.index

        final_time = pd.Timestamp(normal_data.index[-1])
        if freq == 'H':
            initial_datetime = final_time-timedelta(hours=len(normal_data)-1)
        elif freq.__contains__('H'):
            multiplier = int(freq.replace('H', ""))
            initial_datetime = final_time-timedelta(hours=(len(normal_data)-1)*multiplier)
        elif freq.__contains__('T'):
            multiplier = int(freq.replace('T', ""))
            print(final_time)
            initial_datetime = final_time-timedelta(minutes=(len(normal_data)-1)*multiplier)  

        index = pd.date_range(start=initial_datetime, end=final_time, freq=freq)

        model_data = pd.Series(normal_data.values, index=index)
        model_data.columns = ['Close']
        model_data = model_data.asfreq(freq)
        
        return model_data, normal_index
    
    def model_to_normal(self, model_data, normal_index) -> pd.Series:

        normal_data = pd.Series(model_data.values, index=normal_index)
        normal_data.columns = ['Close']
        
        return normal_data
    def log_returns(self, data):
        returns = np.log(data/data.shift(1))
        return returns

