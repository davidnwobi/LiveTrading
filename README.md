
# Live Trading

## Introduction
Greetings! Allow me to guide you through a project I undertook over the summer. The objective was to create a streamlined framework for launching a trading strategy into the market.

*Full Disclaimer: The design for this project wasn't my own creation; I adapted it from source code in the book "Successful Algorithmic Trading" by Michael L. Halls-Moore. This book is an invaluable resource for anyone entering the realm of algorithmic trading.*

## Overview

The project consists of three main components: The Strategy, the Portfolio, and the Execution Handler. Each plays a pivotal role in the success of this framework.

**The Strategy:** An algorithm determining the optimal moments to buy or sell.

**The Portfolio:** The core of the framework deciding on transaction volumes and keeping track of current positions.

**The Execution Handler:** Our executor, translating decisions into real orders. It typically serves as an interface to the broker's API.

Complementing these components are the data retrieval module and the events queue. The data retrieval module is responsible for keeping the data current and operates independently of the rest of the framework. Communication between all four components occurs via the events queue.

## A Sample Workflow
For this sample, I'll use a strategy developed earlier this year. Imagine blending an ARMA model with an AROON indicator. ARMA predicts the next day's price, and the AROON indicator determines whether to buy or sell. In this case, the execution handler is a simulated broker for testing purposes, rather than a real broker. You can find the code for the strategy [here](main.py).

### Define Constants
```python
DATA_PATH = 'D:/Data'
CURRENCY = 'SPX500_USD'
INITIAL_DATA_POINTS = 403

```

### Setup the strategy
```python
# Event queue
events = q.Queue()

data_gen = data_generator(data, INITIAL_DATA_POINTS)
current_data = next(data_gen)
TSM1 = TradingStratsMark1(events, CURRENCY, current_data, 32, 188)
sim_execution_handler = SimulatedExecutionHandler(events, data, 100000)
portfolio = Portfolio(events, sim_execution_handler)
```

### Main loop testing on simulated data

In a real market case, a continuous while loop would suffice, but here, I am running it for a set number of iterations on downloaded data.

We first notify the strategy of the new data. The strategy checks if it has a signal to buy or sell. If it does, it will place an order on the queue.
```python
for _ in tqdm(range(0, len(data) - INITIAL_DATA_POINTS - 63000), desc="Progress: "):
    try:
        data_updated = False
        # Notify strategy of new data
        TSM1.generate_signal(MarketEvent())
```

At this point, note that we could already be in a position. If we are, we update the limit order prices in case we have already been filled. We then update the signal. This will determine if we need to place an order.
```python
if not events.empty():
    # Update portfolio
    signal = events.get()
    portfolio.update_limit_order_prices(sim_execution_handler.get_current_price(CURRENCY))
    portfolio.update_signal(signal, sim_execution_handler.get_current_price(CURRENCY))

```
This is where we actually update the data. Brokers can be a bit tricky. Depending on the broker, you might not be able to get the most up-to-date data except you are streaming and collating it yourself. So I've put in a one candle lag to mimic this delay.

```python
if not events.empty():
    # Update data
    current_data = next(data_gen)
    data_updated = True
    TSM1.data = current_data
    sim_execution_handler.data = current_data
```

Then an order can be placed and filled. The execution handler has place the order with the broker, who should ideally then fill the order and send a fill event. The portfolio will update the current positions accordingly.

```python
# Execute order on new data
order = events.get()
sim_execution_handler.execute_order(order, portfolio)

if not events.empty():
    fill: FillEvent = events.get()
    portfolio.update_fill(fill)
```

Within that flow, we've updated the data. We should check if your limit has been filled. 

```python
portfolio.update_limit_order_prices(sim_execution_handler.get_current_price(CURRENCY))
# Check if limit order has been filled
signal = SignalEvent('TSM1', CURRENCY, current_data.index[-1], "None", 0)
portfolio.update_signal(signal, sim_execution_handler.get_current_price(CURRENCY))
```

If it has, we would need to perform the above flow again. If not, we can just continue.

Also, suppose we did not get a signal to buy or sell, We'll still need to update data and still perform the flow again.

## Improvements to be made

Reflecting on the project a few months later, there are many opportunity for improving code quality.

**Modular construction:** The main script lacks modularity. This creates a situation where there is a lot of duplicated code, and the main script has too much responsibility. A promising path for improvement would be to implement the observer pattern. The data retriever and the event queue could become their own classes. Then, the event queue could play the role of a subject obsserved by the data retriever along with the strategy, portfolio and execution handler. And all the logic could be modularized in there.

**Concurrency:** Some parts of the code can and should be run concurrently.  A case in point is the data retrieval process. It is a distinct and independent operation during online execution. In fact it is run in a separate script. It can be run concurrently with the rest of the code, improving efficiency and actually providing a way to stop it😅.