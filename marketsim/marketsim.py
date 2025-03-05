""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
Student Name: Longtai Liao
GT User ID: lliao32
GT ID:  903648350		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import os  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt

def author():
    return "lliao32"

def study_group():
    return "lliao32"


def compute_portvals(
    orders_file="./orders/orders.csv",  		  	   		 	 	 			  		 			     			  	 
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
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	 	 			  		 			     			  	 
    # TODO: Your code here


    order = pd.read_csv(orders_file, index_col='Date', dtype='|str, str, str,  i4',  parse_dates=True, na_values=['nan']) # todo: may need to check if isinstance()
    order.sort_index()

    start_date = order.index[0]
    end_date = order.index[-1]
    symbols = order['Symbol'].unique()

    prices_record = get_data(symbols, pd.date_range(start_date, end_date)) #stock prices_record of the selected symbols
    prices_record['value'] = pd.Series(0.0, index=prices_record.index) # placeholder for the value of the portfolio

    portfolio = {} # store tfhe number of shares of each stock in the portfolio, and cash
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

    # print(prices_record['value'])
    return prices_record['value']

def get_daily_returns(df):
    daily_return = df.pct_change()
    return daily_return.iloc[1:]

def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./orders/orders-short.csv"
    # of = "./orders/orders-11.csv"

    sv = 1000000  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio stats
    order = pd.read_csv(of, index_col='Date', dtype='|str, str, str,  i4',  parse_dates=True, na_values=['nan']) # todo: may need to check if isinstance()
    order.sort_index()

    start_date = order.index[0]
    end_date = order.index[-1]

    cum_ret = portvals[-1] / portvals[0] - 1  # cumulative_return
    daily_return = get_daily_returns(portvals)
    avg_daily_ret = daily_return.mean()  # avg daily return
    std_daily_ret = daily_return.std()  # standard daily return
    sharpe_ratio = (avg_daily_ret / std_daily_ret) * np.sqrt(252)  # sharp ratio


    prices_SPY = get_data(['$SPX'], pd.date_range(start_date, end_date))
    normalized_SPY = prices_SPY / prices_SPY.iloc[0] * 1000000

    cum_ret_SPY = normalized_SPY.iloc[-1] / normalized_SPY.iloc[0] - 1  # Use iloc for both first and last values
    daily_return_SPY = get_daily_returns(normalized_SPY)
    avg_daily_ret_SPY = daily_return_SPY.mean()  # avg daily return
    std_daily_ret_SPY = daily_return_SPY.std()  # standard daily return
    sharpe_ratio_SPY = (avg_daily_ret_SPY / std_daily_ret_SPY) * np.sqrt(252)  # sharp ratio


    # df_temp = pd.concat(
    #     [portvals, normalized_SPY], keys=["Portfolio", "SPY"], axis=1
    # )
    # df_temp.plot(title="Daily Portfolio Value and SPY")
    # plt.xlabel("Date")
    # plt.ylabel("Normalized Price")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.savefig('Figure1.png')
  		  	   		 	 	 			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
