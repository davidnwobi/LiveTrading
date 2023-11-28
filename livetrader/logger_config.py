import logging
import oandapyV20


def setup_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


flow_logger = setup_logger('flow_logger', 'flow.log')

oandapyV20_logger = setup_logger('oandapyV20', 'oandapyV20.log')

order_logger = setup_logger('order_logger', 'orders.log')

data_retrival_logger = setup_logger('data_retrival_logger', 'data_retrival.log')

sim_flow_logger = setup_logger('sim_flow_logger', 'sim_flow.log')
