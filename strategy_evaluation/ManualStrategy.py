
import datetime as dt
import util as ut
import pandas as pd
import indicators as ind
class ManualStrategy:

    def __int__(self, verbose=False, impact=0.00, commission=0.00):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
    def author(self):
        return "lliao32"

    def study_group(self):
        return "lliao32"

    def add_evidence(self, symbol, sd, ed, sv):
        pass



    # generate orders based on manual strategy
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=1000000):
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.sv = sv

        period = pd.date_range(sd, ed)
        stock_prices = ut.get_data([symbol], period, False)
        stock_prices = stock_prices.ffill().bfill()

        # create orders
        orders = pd.DataFrame(index=stock_prices.index, columns=['Symbol', 'Order', 'Shares'])
        orders['Symbol'] = symbol
        orders['Order'] = ''
        orders['Shares'] = 0

        self.holding=0

        #  indicator used : MACD, RSI, BBP
        macd = ind.macd(stock_prices[symbol])
        rsi = ind.rsi(stock_prices[symbol])
        bbp = ind.bbp(stock_prices[symbol])

        trades = self.manual_strategy(macd, rsi, bbp)
        return trades

    def manual_strategy(self, macd, rsi, bbp):


        trades = macd.copy()
        trades.loc[:] = 0


        # Create a new column for the symbol in orders dataframe

        # Track position
        position = 0
        signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()

        # Iterate through each trading day
        for date in trades.index:
            # # Skip if date not in indicator data
            # if date not in macd.index or date not in rsi.index or date not in bbp.index:
            #     continue

            # Get current indicator values
            cur_macd = macd.loc[date]
            cur_macd_signal = signal.loc[date]
            cur_rsi = rsi.loc[date]
            cur_bbp = bbp.loc[date]

            # Define buy and sell signals based on indicators
            # Buy signal: MACD above signal line, RSI < 30 (oversold), BBP < 0 (oversold)
            buy_signal = (cur_macd > cur_macd_signal) and (cur_rsi < 30 or cur_bbp < 0)

            # Sell signal: MACD below signal line, RSI > 70 (overbought), BBP > 1 (overbought)
            sell_signal = (cur_macd < cur_macd_signal) and (cur_rsi > 70 or cur_bbp > 1)

            # Apply trading logic based on current position
            if position == -1000:  # Currently short
                if buy_signal:
                    trades.loc[date] = 2000
                    position = 1000
            elif position == 0:  # No position
                if buy_signal:
                    trades.loc[date] = 1000 # Go long

                    position = 1000
                elif sell_signal:
                    # orders.at[date, 'Shares'] = -1000
                    trades.loc[date] = -1000 # Go short

                    position = -1000
            elif position == 1000:  # Currently long
                if sell_signal:
                    # orders.at[date, 'Shares'] = -2000
                    trades.loc[date] = -2000 # Sell and go short

                    position = -1000

        return trades
