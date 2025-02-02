""""""
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
GT ID: 903648350			  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data
import scipy.optimize as opt

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "lliao32"  # replace tb34 with your Georgia Tech username.

def study_group():
    return "lliao32"

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903648350  # replace with your GT ID number
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality
def optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			     			  	 
    gen_plot=False,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			     			  	 
    statistics.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
    :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
    :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			     			  	 
        symbol in the data directory)  		  	   		 	 	 			  		 			     			  	 
    :type syms: list  		  	   		 	 	 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	 	 			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	 	 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			     			  	 
    :rtype: tuple  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			     			  	 
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		 	 	 			  		 			     			  	 


    allocs = find_optimal_allocs(prices)
    # print("### res: ", allocs)

    port_val = get_port_val(allocs, prices)

    # compute stats: cr, adr, sddr, sr
    cr = port_val[-1] / port_val[0] -1 # cumulative_return
    daily_return = get_daily_returns(port_val)
    adr = daily_return.mean() # avg daily return
    sddr = daily_return.std() # standard daily return
    sr = (adr / sddr) * np.sqrt(252) # sharp ratio

  		  	   		 	 	 			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			     			  	 
    if gen_plot:
        # add code to plot here
        normalized_SPY = prices_SPY / prices_SPY.iloc[0]
        df_temp = pd.concat(  		  	   		 	 	 			  		 			     			  	 
            [port_val, normalized_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp.plot(title="Daily Portfolio Value and SPY")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig('Figure1.png')

    return allocs, cr, adr, sddr, sr

def find_optimal_allocs(prices):
    stock_num = prices.shape[1]
    init_val = 1 / stock_num
    init_allocs = np.asarray([init_val for i in range(stock_num)])
    bounds = [(0, 1) for i in range(stock_num)]
    # set constrains that all alloc sum up to 1
    constraints = {'type': 'eq', 'fun': lambda allocs: np.sum(allocs) - 1.0}
    solution = opt.minimize(
        sharp_ratio_ng,
        init_allocs,
        args=(prices,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return solution.x


def sharp_ratio_ng(allocs, prices):
    port_val = get_port_val(allocs, prices)
    # print("### port val: ", port_val)
    daily_return = get_daily_returns(port_val)
    # print("### daily return: ", daily_return)
    # Compute Sharpe ratio
    avg_daily_return = daily_return.mean()  # Expected return
    std_daily_return = daily_return.std()  # Risk (volatility)

    ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) # 252 trading days
    # print("-- ", allocs, ratio)
    return ratio * (-1)

def get_port_val(allocs, prices):
    normalized_stock = prices / prices.iloc[0, :]
    allocation_portfolio = normalized_stock * allocs # normalize price then multiply allocation
    # flatten the portfolio , merge all stocks into one column
    port_val = allocation_portfolio.sum(axis = 1)

    return port_val

def get_daily_returns(df):
    daily_return = df.pct_change()
    return daily_return.iloc[1:]

# def test_code():
#     """
#     This function WILL NOT be called by the auto grader.
#     """
#
#     start_date = dt.datetime(2008, 1, 1)
#     end_date = dt.datetime(2009, 1, 1)
#     symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]
#
#     # Assess the portfolio
#     allocations, cr, adr, sddr, sr = optimize_portfolio(
#         sd=start_date, ed=end_date, syms=symbols, gen_plot=True
#     )
#
#     # Print statistics
#     print(f"Start Date: {start_date}")
#     print(f"End Date: {end_date}")
#     print(f"Symbols: {symbols}")
#     print(f"Allocations:{allocations}")
#     print(f"Sharpe Ratio: {sr}")
#     print(f"Volatility (stdev of daily returns): {sddr}")
#     print(f"Average Daily Return: {adr}")
#     print(f"Cumulative Return: {cr}")

def test_figure_1():
    print("============== figure 1 ==============")
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")

  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	 	 			  		 			     			  	 
    # test_code()
    test_figure_1()
