import pandas as pd
import datetime as dt
import util as ut

class BenchmarkStrategy:
    def __init__(self, verbose=False, impact=0.00, commission=0.00):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def author(self):
        return "lliao32"

    def study_group(self):
        return "lliao32"

    def add_evidence(self, symbol, sd, ed, sv):
        pass


    # generate benchmark orders
    def benchmarkOrder(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=1000000):
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

        orders.iloc[0] = [symbol, "BUY", 1000]
        orders.iloc[-1] = [symbol, "SELL", 1000]
        print("bench mark ==== orders")
        print(orders)


        if self.verbose:
            print(orders)
        return orders

