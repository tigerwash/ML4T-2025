import ManualStrategy
import marketsimcode as mktsim
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import StrategyLearner
import util as ut
def author():
    return "lliao32"

def study_group():
    return "lliao32"


# create orders from trades
    """
    Process the trades dataframe and convert it to orders format

    :param trades: DataFrame with a single column containing trade signals (+1000, -1000, etc.)
    :param selected_symbol: The stock symbol being traded
    :return: DataFrame with columns ['Symbol', 'Order', 'Shares']
    """
def process_orders(trades, selected_symbol):
    orders = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders['Symbol'] = selected_symbol
    for date, row in trades.iterrows():
        trade_value = row.iloc[0]  # Get the trade value (assuming single column DataFrame)

        if trade_value > 0:
            orders.loc[date, 'Order'] = 'BUY'
            orders.loc[date, 'Shares'] = abs(trade_value)
            # print(f"BUY: {date} {selected_symbol} {abs(trade_value)}")
        elif trade_value < 0:
            orders.loc[date, 'Order'] = 'SELL'
            orders.loc[date, 'Shares'] = abs(trade_value)
            # print(f"SELL: {date} {selected_symbol} {abs(trade_value)}")
        else:
            orders.loc[date, 'Order'] = ''
            orders.loc[date, 'Shares'] = 0
    return orders

# todo: move to manual_strategy.py
def manual_strategy_runner(symbol, sd, ed, sv, out_sample = False):

    ms = ManualStrategy.ManualStrategy()
    trades = ms.testPolicy(symbol, sd, ed, sv)
    # print("manual strategy ==== trades")
    orders = process_orders(trades, symbol)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=9.95, impact=0.005)

    return value

#  generate benchmark performance result
def benchmark_runner(symbol, sd, ed, sv):

    orders = benchmarkOrder(symbol, sd, ed, sv)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=0.0, impact=0.0)

    return value

def strategy_learner_runner(symbol, sd, ed, sv):
    sl = StrategyLearner.StrategyLearner()
    # in sample training: 2008-01-01 to 2009-12-31
    sl.add_evidence(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 1000000)

    trades = sl.testPolicy(symbol, sd, ed, sv)
    # print("strategy learner ==== trades")
    orders = process_orders(trades, symbol)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=9.95, impact=0.005)
    # value = mktsim.compute_portvals(orders, start_val=sv, commission=0, impact=0.000)

    return value
def experiment1_plot(symbol, sd, ed, sv, in_sample = True):
    dates = pd.date_range(sd, ed)
    df_container = pd.DataFrame(index=dates)

    ms_result = manual_strategy_runner(symbol, sd, ed, sv)
    bm_result = benchmark_runner(symbol, sd, ed, sv)
    sl_result = strategy_learner_runner(symbol, sd, ed, sv)

    df_container["Manual_strategy"] = normalize(ms_result)
    df_container["Benchmark_strategy"] = normalize(bm_result)
    df_container["Strategy_learner"] = normalize(sl_result)

    sample_statue = "In-Sample" if in_sample else "Out-of-Sample"

    # Get trade entry points for plotting vertical lines
    # long_entries = ms_orders.index[ms_orders[symbol] > 0]  # Dates where you go long_entries
    # short_entries = ms_orders.index[ms_orders[symbol] < 0]  # Dates where you go short

    # Plot the result
    plt.figure(figsize=(12, 6))
    plt.plot(df_container.index, df_container["Manual_strategy"], 'r-', label="Manual Strategy")
    plt.plot(df_container.index, df_container["Benchmark_strategy"], 'purple', label="Benchmark")
    plt.plot(df_container.index, df_container["Strategy_learner"], 'b-', label="Strategy Learner")
    plt.legend(loc='best')  # 'best' places the legend in the optimal location to avoid covering data

    plt.title(f"Manual Strategy vs Benchmark Strategy - {symbol} - {sample_statue}", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot (required for grading)
    plt.tight_layout()
    plt.savefig(f'images/experiment1_{sample_statue}.png')  # Save the plot to a file
    # plt.show(block=True)

    return df_container

def benchmarkOrder(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=1000000):

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

    return orders
def normalize(df):
    return df / df.iloc[0]

if __name__ == "__main__":
    symbol = "JPM"
    # symbol = "AAPL"
    # symbol = "SINE_FAST_NOISE"
    # symbol = "AIV"
    # symbol = "ML4T-220"
    # sd = dt.datetime(2010, 1, 1)
    # ed = dt.datetime(2011, 12, 31)
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 1000000

    # ms = manual_strategy_runner(symbol, sd, ed, sv)

    experiment1_plot(symbol, sd, ed, sv)
