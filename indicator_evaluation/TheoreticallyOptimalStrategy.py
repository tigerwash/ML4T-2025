import datetime as dt

import pandas as pd
import marketsimcode as mktsim
from util import get_data
import matplotlib.pyplot as plt


def author():
    return "lliao32"

def study_group():
    return "lliao32"


def testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    symbols = [symbol]
    exame_date = pd.date_range(sd, ed)
    dated_prices = get_data(symbols, exame_date, False)
    dated_prices = dated_prices.ffill().bfill()
    single_stock = dated_prices[symbol]
    # single_stock = single_stock.ffill().bfill()
    dates = single_stock.index
    trades = single_stock.copy()
    trades.loc[:] = 0

    holding = 0
    for i in range(0, len(dates)-1):
        cur_price = single_stock.loc[dates[i]]
        next_price = single_stock.loc[dates[i+1]]
        if cur_price < next_price:
            trade = 1000 - holding
        else:
            trade = -1000 - holding

        trades.loc[dates[i]] = trade
        holding += trade
    trades = trades.to_frame(name='Shares')
    return trades


def benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    order = pd.DataFrame({
        'Symbol': ['JPM', 'JPM'],
        'Order': ['BUY', 'SELL'],
        'Shares': [1000, 1000]
    }, index=pd.DatetimeIndex([sd, ed]))
    order.index.name = 'Date'
    benchmark_vals = mktsim.compute_portvals(order, start_val=sv, commission=0, impact=0)

    return benchmark_vals

# def performance_table(benchmark, portfolio):
#     # Cumulative return
#     cr_ben = benchmark.iloc[-1] / benchmark.iloc[0] - 1
#     cr_port = portfolio.iloc[-1] / portfolio.iloc[0] - 1
#
#     # Stdev of daily returns of benchmark and portfolio
#     std_ben = benchmark.pct_change().std()
#     std_port = portfolio.pct_change().std()
#
#     # Mean of daily returns of benchmark and portfolio
#     mean_ben = benchmark.pct_change().mean()
#     mean_port = portfolio.pct_change().mean()
#
#     print("************ performance_table *******************")
#     print("Cumulative return of benchmark: ", cr_ben)
#     print("Cumulative return of portfolio: ", cr_port)
#     print("Stdev of daily returns of benchmark: ", std_ben)
#     print("Stdev of daily returns of portfolio: ", std_port)
#     print("Mean of daily returns of benchmark: ", mean_ben)
#     print("Mean of daily returns of portfolio: ", mean_port)
#     print("************ performance_table *******************")

def performance_table(benchmark_portvals, tos_portvals):
    # Cumulative Return
    cum_return_ben = benchmark_portvals['Portfolio Val'][-1]
    cum_return_tos = tos_portvals['Portfolio Val'][-1]

    # daily return percentage
    d_ben = (benchmark_portvals / benchmark_portvals.shift(1) -1)
    d_tos = (tos_portvals / tos_portvals.shift(1) -1)

    d_ben = d_ben.iloc[1:]
    d_tos = d_tos.iloc[1:]

    # Stdev
    std_ben = d_ben.std()
    std_tos = d_tos.std()

    # mean
    m_ben = d_ben.mean()
    m_tos = d_tos.mean()

    print("############## Performance Metrics ################")
    print("--")
    print("*** Cumulative Returns:")
    print("benchmark:" + str(cum_return_ben))
    print("tos:" + str(cum_return_tos))
    print("--")
    print("*** Standard Deviation of Daily Returns:")
    print("benchmark:" + str("%.4f" % round(std_ben, 4)))
    print("tos:" + str("%.4f" % round(std_tos, 4)))

    print("--")
    print("*** Mean of Daily Returns in percentage:")
    print("benchmark:" + str("%.4f" % round(m_ben, 4)))
    print("tos:" + str("%.4f" % round(m_tos, 4)))
    print("")
    print("############## ****************** ################")




def chart_plot(benchmark, portfolio):
    plt.figure(figsize=(10, 5))
    plt.plot(benchmark, label='Benchmark', color="purple")
    plt.plot(portfolio, label='Portfolio', color="red")
    plt.title('Theoretically Optimal Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('TOS_plot.png')


def TOS_plot():

    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades = mktsim.prepare_order_file(df_trades, symbol="JPM")
    JPM_folio = mktsim.compute_portvals(df_trades, start_val=100000, commission=0, impact=0)
    benchmark_folio = benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    performance_table(benchmark_folio, JPM_folio)

    # normalize the portfolio and benchmark, starting with 1
    benchmark_norm = benchmark_folio / benchmark_folio.iloc[0]
    JPM_folio_norm = JPM_folio / JPM_folio.iloc[0]

    chart_plot(benchmark_norm, JPM_folio_norm)
