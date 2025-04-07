import ManualStrategy
import marketsimcode as mktsim
import datetime as dt
import BenchmarkStrategy
import pandas as pd
import matplotlib.pyplot as plt
def author():
    return "lliao32"

def study_group():
    return "lliao32"


# create orders from trades
def process_orders(trades, selected_symbol):
    orders = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])
    orders['Symbol'] = selected_symbol
    for i in range(len(trades)):
        if trades.iloc[i] > 0:
            orders.iloc[i] = [selected_symbol, "BUY", trades.iloc[i]]
        elif trades.iloc[i] < 0:
            orders.iloc[i] = [selected_symbol, "SELL", abs(trades.iloc[i])]
        else:
            orders.iloc[i] = [selected_symbol, "", 0]
    return orders
def manual_strategy_runner(symbol, sd, ed, sv, out_sample = False):

    ms = ManualStrategy.ManualStrategy()
    trades = ms.testPolicy(symbol, sd, ed, sv)
    orders = process_orders(trades, symbol)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=0.0, impact=0.0)

    return value

#  generate benchmark performance result
def benchmark_runner(symbol, sd, ed, sv):

    bm = BenchmarkStrategy.BenchmarkStrategy()
    orders = bm.benchmarkOrder(symbol, sd, ed, sv)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=0.0, impact=0.0)

    return value
def experiment1_plot(symbol, sd, ed, sv):
    dates = pd.date_range(sd, ed)
    df_container = pd.DataFrame(index=dates)

    ms_result = manual_strategy_runner(symbol, sd, ed, sv)
    bm_result = benchmark_runner(symbol, sd, ed, sv)

    df_container["Manual_strategy"] = normalize(ms_result)
    df_container["Benchmark_strategy"] = normalize(bm_result)


    # Get trade entry points for plotting vertical lines
    # long_entries = ms_orders.index[ms_orders[symbol] > 0]  # Dates where you go long_entries
    # short_entries = ms_orders.index[ms_orders[symbol] < 0]  # Dates where you go short

    # Plot the result
    plt.figure(figsize=(12, 6))
    plt.plot(df_container.index, df_container["Manual_strategy"], 'r-', label="Manual Strategy")
    plt.plot(df_container.index, df_container["Benchmark_strategy"], 'purple', label="Benchmark")

    # Add vertical lines for entry points (as per requirements)
    # for date in long_entries:
    #     plt.axvline(date, color='blue', linestyle='--', alpha=0.5)
    #
    # for date in short_entries:
    #     plt.axvline(date, color='black', linestyle='--', alpha=0.5)

    plt.title(f"Manual Strategy vs Benchmark Strategy - {symbol}", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot (required for grading)
    plt.savefig(f"experiment1_{symbol}.png")
    plt.tight_layout()
    plt.show()

    # Calculate and display performance metrics (required for report)
    # ms_daily_returns = ms_portvals.pct_change().dropna()
    # bm_daily_returns = bm_portvals.pct_change().dropna()
    #
    # ms_cum_return = ms_portvals.iloc[-1] / ms_portvals.iloc[0] - 1
    # bm_cum_return = bm_portvals.iloc[-1] / bm_portvals.iloc[0] - 1

    print(f"Performance Metrics for {symbol} ({sd.strftime('%Y-%m-%d')} to {ed.strftime('%Y-%m-%d')}):")
    print(f"{'Metric':<20} {'Manual Strategy':<20} {'Benchmark':<20}")
    print(f"{'-' * 60}")
    # print(f"{'Cumulative Return':<20} {ms_cum_return:>18.4f} {bm_cum_return:>18.4f}")
    # print(f"{'Std Daily Returns':<20} {ms_daily_returns.std():>18.4f} {bm_daily_returns.std():>18.4f}")
    # print(f"{'Mean Daily Returns':<20} {ms_daily_returns.mean():>18.4f} {bm_daily_returns.mean():>18.4f}")

    return df_container


def normalize(df):
    return df / df.iloc[0]

if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 1000000

    # ms = manual_strategy_runner(symbol, sd, ed, sv)

    experiment1_plot(symbol, sd, ed, sv)
