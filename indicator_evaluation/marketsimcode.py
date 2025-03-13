import pandas as pd
from util import get_data
def author():
    return "lliao32"

def study_group():
    return "lliao32"


def prepare_order_file(trades, symbol):
    """
    Prepare the order file for the trades

    :param trades: the trades to be made
    :type trades: pandas.DataFrame
    :return: the order file
    :rtype: pandas.DataFrame
    """

    trades['Symbol'] = symbol
    trades['Order'] = 'BUY'
    dates = trades.index
    for i in range(len(dates) - 1):
        d = dates[i]
        trade = trades.loc[d].loc['Shares']
        order_type = 'SELL' if trade < 0 else 'BUY'
        trades.at[d, 'Order'] = order_type
        trades.at[d, 'Shares'] = abs(trade)
    return trades
def compute_portvals(
        order,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    order.sort_index()

    start_date = order.index[0]
    end_date = order.index[-1]
    symbols = order['Symbol'].unique()

    prices_record = get_data(symbols,
                             pd.date_range(start_date, end_date), False)  # stock prices_record of the selected symbols
    prices_record = prices_record.ffill().bfill()
    prices_record['value'] = pd.Series(0.0, index=prices_record.index)  # placeholder for the value of the portfolio

    portfolio = {}  # store tfhe number of shares of each stock in the portfolio, and cash
    for s in symbols:
        portfolio[s] = 0
    portfolio['cash'] = start_val

    for date, price in prices_record.iterrows():
        for index, od in order.iterrows():
            if index == date:
                if od['Order'] == 'BUY':
                    portfolio[od['Symbol']] += od['Shares']
                    portfolio['cash'] -= od['Shares'] * price[od['Symbol']] * (1 + impact) + commission
                elif od['Order'] == 'SELL':
                    portfolio[od['Symbol']] -= od['Shares']
                    portfolio['cash'] += od['Shares'] * price[od['Symbol']] * (1 - impact) - commission

        # Calculate total stock value
        stock_value = 0
        for symbol in symbols:
            shares = portfolio[symbol]
            stock_price = price[symbol]
            stock_value += shares * stock_price

        # Add cash and update the value
        total_value = stock_value + portfolio['cash']
        prices_record.loc[date, 'value'] = total_value

    return prices_record['value']