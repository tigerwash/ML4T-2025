import datetime as dt
import indicators as ind
import RTLearner as rtl
import BagLearner as bl
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: longtai liao		  	   		 	 	 			  		 			     			  	 
GT User ID: lliao32  	   		 	 	 			  		 			     			  	 
GT ID: 903648350 	  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import random  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
import util as ut  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(rtl.RTLearner, kwargs={"leaf_size": 10}, bags=20, boost=False, verbose=False)
        # self.learner = rtl.RTLearner(leaf_size=1, verbose=False)
  		  	   		 	 	 			  		 			     			  	 
    # this method should create a QLearner, and train it for trading

    def author(self):
        return 'lliao32'

    def study_group(self):
        return "lliao32"

    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # add your code to do learning here  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # example usage of the old backward compatible util function  		  	   		 	 	 			  		 			     			  	 
        syms = [symbol]  		  	   		 	 	 			  		 			     			  	 
        period = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, period, False)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices = prices.ffill().bfill()


        train_x = self.indicators(prices, symbol)
        train_y = self.construct_y_data(prices, symbol)

        self.learner.add_evidence(train_x.values, train_y.values)  # add training data to learner


    #  construct data_y for training, the y_data is the action we need to take: 1 buy, -1 sell, 0 hold
    def construct_y_data(self, prices, symbol):
        dates = prices.index
        trade_times = len(dates)
        actions = pd.Series(0.0, index=prices.index, name=symbol)

        cur_holding = 0

        for i in range(trade_times -1):
            cur_price = prices[symbol].loc[dates[i]]
            next_price = prices[symbol].loc[dates[i + 1]]

            if cur_price < next_price:
                trade = 1000 - cur_holding # price up, buy
            else:
                trade = -1000 - cur_holding # price down, sell
            actions.loc[dates[i]] = trade

            cur_holding += trade

        y_data = pd.DataFrame(actions)

        # normalize the y_data
        for index in y_data.index:
            if y_data.loc[index, symbol] > 0:
                y_data.loc[index, symbol] = 1
            elif y_data.loc[index, symbol] < 0:
                y_data.loc[index, symbol] = -1
            else:
                y_data.loc[index, symbol] = 0

        return y_data # or return y_data.values


    # this method should use the existing policy and test it against new data
    # return the trades based on the learner's prediction
    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # here we build a fake set of trades  		  	   		 	 	 			  		 			     			  	 
        # your code should return the same sort of data  		  	   		 	 	 			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices_all = prices_all.ffill().bfill()
        # trades = prices_all[[symbol,]]  # only portfolio symbols


        train_x = self.indicators(prices_all, symbol)

        predictions = self.learner.query(train_x.values)  # add training data to learner , train_x or values?

        print("==================== predictions")
        print(predictions)

        # Create trades DataFrame
        trades = pd.DataFrame(0, index=train_x.index, columns=[symbol])

        # Convert predictions to trades
        position = 0
        for i, date in enumerate(train_x.index):
            pred = predictions[i]

            # If predict buy (1)
            if pred > 0 and position <= 0:
                action = 1000 - position  # Buy to long position
                trades.loc[date, symbol] = action
                position = 1000

            # If predict sell (-1)
            elif pred < 0 and position >= 0:
                action = -1000 - position  # Sell to short position
                trades.loc[date, symbol] = action
                position = -1000

        print("==================== trades")
        print(trades)
        return trades

    # get the indicators for the given stock prices, used for training the learner
    def indicators(self, prices, symbol):
        macd = ind.macd(prices[symbol])
        macd_signal = ind.macd_signal(macd)
        macd_hist = ind.macd_hist(macd, macd_signal)


        rsi = ind.rsi(prices[symbol])
        bbp = ind.bbp(prices[symbol])
        # sma = ind.sma(prices[symbol])
        indicator_matrix = pd.concat([macd_hist, rsi, bbp], axis=1)
        indicator_matrix.columns = ['MACD_hist', 'RSI', 'BBP']
        # indicator_matrix = indicator_matrix.dropna()
        indicator_matrix = indicator_matrix.fillna(0)
        print("==================== indicator_matrix")
        print(indicator_matrix)
        return indicator_matrix

  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":
    sl = StrategyLearner()
    add_evidence = st.add_evidence("JPM", dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 1000000)
    out = sl.testPolicy("JPM", dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 1000000)
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
