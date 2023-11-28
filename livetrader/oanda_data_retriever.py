import pathlib
from datetime import datetime, timedelta
import time
import tpqoa
from dotenv import load_dotenv, dotenv_values
from .logger_config import flow_logger, oandapyV20_logger, data_retrival_logger

load_dotenv(dotenv_path="../vars.env")

if __name__ == "__main__":
    while True:
        def calculate_start_time(timeframe: str, end_time: datetime, time_periods: int):
            valid_timeframes = ['H', 'M', 'S']

            if timeframe[-1] not in valid_timeframes:
                raise ValueError('Invalid timeframe')

            try:
                multiplier = int(timeframe[:-1])
            except ValueError:
                raise ValueError('Invalid timeframe')

            if timeframe[-1] == 'H':
                timedelta_value = timedelta(hours=time_periods * multiplier)
            elif timeframe[-1] == 'M':
                timedelta_value = timedelta(minutes=time_periods * multiplier)
            else:
                timedelta_value = timedelta(seconds=time_periods * multiplier)

            start_datetime = end_time - timedelta_value

            return start_datetime


        def get_data(symbol="SPX500_USD", freq="1H", time_periods=188, path=pathlib.Path.cwd()):
            oa_freq = freq[::-1]
            oanda = tpqoa.tpqoa("../private_stuff/oanda.cfg")
            i = 0
            required_time_periods = time_periods
            while True:
                try:
                    start_datetime = calculate_start_time(freq, datetime.now(), time_periods)
                except ValueError:
                    raise ValueError('Invalid timeframe')
                data = oanda.get_history(instrument=symbol,
                                         start=start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                         end=(datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                                         granularity=oa_freq,
                                         price="M",
                                         localize=False).tz_convert("Europe/London")
                if len(data) >= required_time_periods:
                    try:
                        directory_path = pathlib.Path.cwd() / symbol
                        directory_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
                        file_path = directory_path / f'{symbol}_{datetime.now().strftime("%H")}.csv'
                        data.to_csv(file_path)
                        data_retrival_logger.info(f"New Data Available: Successfully saved to {file_path}")
                        flow_logger.info(f"New Data Available: Successfully saved to {file_path}")
                        break
                    # catch any errors
                    except Exception as e:
                        flow_logger.error(f"Error: {e}")
                        data_retrival_logger.error(f"Error: {e}")
                        print(e)

                time.sleep(5)
                time_periods += 100
                if i > 50:
                    break
                i += 1


        if datetime.now().minute == 45:
            print("Getting data...")
            get_data(symbol="AUD_USD", freq="1H", time_periods=188)
            print("Done")
            time.sleep(60 * 60 - 60)

