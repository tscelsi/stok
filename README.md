# Stok

## Framework for running investing strategy simulations.

Predict buying & selling actions given the day's opening stock price.

For a particular stock:
    - historical data is downloaded from Yahoo Finance (yfinance package)
    - current day opening stock price

This information is passed to a strategy module which returns a buy/sell action.

action object:
    - buy | sell | hold
    - quantity
    - price
    - date
    - stock
    - confidence


Simple MVP:

For one stock (GOOG)
1. get historical data and put into a context class
2. have some simple investing strategy that takes context and returns the above action object
3. build simulator that simulates the strategy over a given time period and can provide some analytics on the results



give the strategy larger and smaller context windows


what sort of strategies?

- fixed strategy that looks at the last 5 days and buys if the price is lower than the average of the last 5 days



- Context class dictates what context is available to any given strategy
- Strategy class dictates basically what to do with the context
- Simulator class is able to run some number of days of simulations and provide some analytics on the results




```python
s = Simulator("path/to/config.yaml")
s.run()
```

config:
```yaml

```


When I invest in stocks I start with a fixed sum of money which I can't exceed when buying stocks. If I want to buy more stocks I can sell some of my existing stocks. I can't sell more stocks than I have.

selling:
- I can always sell my stocks


buying:
- I can buy stocks if I have enough money
- I should try to limit my losses


holding:
- I can hold my stocks if I don't want to buy or sell



date         symbol  quantity  unit_price  value
2022-10-09   GOOG    1         1.231       2.462
2022-10-09   TSLA    2         44.58       89.160




TODO:

- train different types of models for a ticker
- train e.g. PPO over all dji tickers and see if overall we make more than the index by trading stocks individually
- try a multi stock training?