
import datetime as dt
import util as ut
import pandas as pd
import indicators as ind
import marketsimcode as mktsim
import matplotlib.pyplot as plt
class ManualStrategy:

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



    # generate orders based on manual strategy
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=1000000):
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.sv = sv

        period = pd.date_range(self.sd, self.ed)
        stock_prices = ut.get_data([self.symbol], period, False)
        stock_prices = stock_prices.ffill().bfill()

        # create orders
        orders = pd.DataFrame(index=stock_prices.index, columns=['Symbol', 'Order', 'Shares'])
        orders['Symbol'] = self.symbol
        orders['Order'] = ''
        orders['Shares'] = 0

        self.holding=0

        #  indicator used : MACD, RSI, BBP
        macd = ind.macd(stock_prices[self.symbol])
        rsi = ind.rsi(stock_prices[self.symbol])
        bbp = ind.bbp(stock_prices[self.symbol])
        mmt = ind.momentum(stock_prices[self.symbol])

        trades = self.manual_strategy(macd, rsi, bbp, mmt)
        return trades

    def manual_strategy(self, macd, rsi, bbp, mmt):


        # trades = macd.copy()
        # trades.loc[:] = 0
        trades = pd.DataFrame(0, index=macd.index, columns=[self.symbol])

        # Create a new column for the symbol in orders dataframe

        # Track position
        position = 0
        macd_signal = ind.macd_signal(macd)

        macd_hist = ind.macd_hist(macd, macd_signal)

        # Skip the first day since we need yesterday's data
        for i in range(1, len(trades.index)):
            # Get current date and previous date
            date = trades.index[i]
            prev_date = trades.index[i - 1]

            # Get PREVIOUS DAY indicator values (this is the key change)
            cur_macd = macd_hist.loc[prev_date]
            cur_macd_signal = macd_signal.loc[prev_date]
            cur_macd_hist = macd_hist.loc[prev_date]
            cur_rsi = rsi.loc[prev_date]
            cur_bbp = bbp.loc[prev_date]
            cur_mmt = mmt.loc[prev_date]

            buy_score = 0
            sell_score = 0

            # MACD - histogram
            if cur_macd_hist > 0.01:
                buy_score += 2
            elif cur_macd_hist < -0.01:
                sell_score += 2
            elif cur_macd_hist > 0:
                buy_score += 1
            elif cur_macd_hist < 0:
                sell_score += 1

            # RSI - overbought/oversold
            if cur_rsi < 30:
                buy_score += 2
            elif cur_rsi < 40:
                buy_score += 1
            elif cur_rsi > 70:
                sell_score += 2
            elif cur_rsi > 60:
                sell_score += 1

            # BBP - 价格波动指标
            if cur_bbp < 0.2:
                buy_score += 2
            elif cur_bbp < 0.4:
                buy_score += 1
            elif cur_bbp > 0.8:
                sell_score += 2
            elif cur_bbp > 0.6:
                sell_score += 1

            # momentum - strength of trend
            if cur_mmt > 0.02:
                buy_score += 1
            elif cur_mmt < -0.02:
                sell_score += 1

            buy_signal = buy_score >= 5
            sell_signal = sell_score >= 5

            # ========================
            # Apply trading logic based on current position
            if position == -1000:  # Currently short
                if buy_signal:
                    trades.loc[date, self.symbol] = 2000
                    position = 1000
            elif position == 0:  # No position
                if buy_signal:
                    trades.loc[date, self.symbol] = 1000 # Go long

                    position = 1000
                elif sell_signal:
                    trades.loc[date, self.symbol] = -1000 # Go short

                    position = -1000
            elif position == 1000:  # Currently long
                if sell_signal:
                    trades.loc[date, self.symbol] = -2000 # Sell and go short

                    position = -1000

        return trades




    def performance_ploter(self, orders, benchmark_values, in_sample = True):
        # Compute portfolio values

        manual_values = mktsim.compute_portvals(orders, start_val=self.sv, commission=self.commission, impact=self.impact)

        self.print_performance_metrics(manual_values, benchmark_values, in_sample)

        df_container = pd.DataFrame(index=benchmark_values.index)

        df_container["Manual_strategy"] = self.normalize(manual_values)
        df_container["Benchmark_strategy"] = self.normalize(benchmark_values)

        sample_statue = "In-Sample" if in_sample else "Out-of-Sample"

        # Get trade entry points for plotting vertical lines
        long_entries = orders.index[orders["Order"] == "BUY"]  # Dates where you go long_entries
        short_entries = orders.index[orders["Order"] == "SELL"]  # Dates where you go short

        plt.figure(figsize=(12, 6))

        # Create one vertical line for buy entries with a label
        if len(long_entries) > 0:
            plt.axvline(long_entries[0], color='blue', linestyle='--', alpha=0.5, label="Buy Entry")
            # Create the rest without labels
            for date in long_entries[1:]:
                plt.axvline(date, color='blue', linestyle='--', alpha=0.5)

        # Create one vertical line for sell entries with a label
        if len(short_entries) > 0:
            plt.axvline(short_entries[0], color='black', linestyle='--', alpha=0.5, label="Sell Entry")
            # Create the rest without labels
            for date in short_entries[1:]:
                plt.axvline(date, color='black', linestyle='--', alpha=0.5)

        plt.plot(df_container.index, df_container["Manual_strategy"], 'r-', label="Manual Strategy")
        plt.plot(df_container.index, df_container["Benchmark_strategy"], 'purple', label="Benchmark")
        plt.legend(loc='best')  # 'best' places the legend in the optimal location to avoid covering data


        plt.title(f"Manual Strategy vs Benchmark {sample_statue}")
        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save the plot (required for grading)
        plt.tight_layout()
        plt.savefig(f"images/Manual_Strategy_{sample_statue}.png")  # Save the plot to a file
        # plt.show(block=True)

    def print_performance_metrics(self, manual_values, benchmark_values, in_sample=True):
        """
        Calculate performance metrics for the benchmark and Manual Strategy,
        print them to console, and save to a CSV file in the images folder.

        Parameters:
        - manual_values: DataFrame with Manual Strategy portfolio values
        - benchmark_values: DataFrame with Benchmark portfolio values
        - in_sample: Boolean indicating if this is for in-sample or out-of-sample period
        """
        import os

        # Create images directory if it doesn't exist
        images_dir = "images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Calculate daily returns
        manual_daily_returns = manual_values.pct_change().dropna()
        benchmark_daily_returns = benchmark_values.pct_change().dropna()

        # Cumulative return
        manual_cum_return = manual_values.iloc[-1] / manual_values.iloc[0] - 1
        benchmark_cum_return = benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1

        # Standard deviation of daily returns - FIX: handle both Series and DataFrame
        if isinstance(manual_daily_returns, pd.DataFrame):
            manual_std = manual_daily_returns.std().iloc[0]
            benchmark_std = benchmark_daily_returns.std().iloc[0]
        else:  # It's a Series
            manual_std = manual_daily_returns.std()
            benchmark_std = benchmark_daily_returns.std()

        # Mean of daily returns - FIX: handle both Series and DataFrame
        if isinstance(manual_daily_returns, pd.DataFrame):
            manual_mean = manual_daily_returns.mean().iloc[0]
            benchmark_mean = benchmark_daily_returns.mean().iloc[0]
        else:  # It's a Series
            manual_mean = manual_daily_returns.mean()
            benchmark_mean = benchmark_daily_returns.mean()

        if self.verbose:
            # Print heading
            print("\n" + "=" * 50)
            print(f"{'In-Sample' if in_sample else 'Out-of-Sample'} Performance Metrics:")
            print("=" * 50)

            # Print metrics
            print(f"Cumulative Return:")
            print(f"  Manual Strategy: {manual_cum_return:.6f}")
            print(f"  Benchmark:      {benchmark_cum_return:.6f}")

            print(f"\nSTDEV of Daily Returns:")
            print(f"  Manual Strategy: {manual_std:.6f}")
            print(f"  Benchmark:      {benchmark_std:.6f}")

            print(f"\nMean of Daily Returns:")
            print(f"  Manual Strategy: {manual_mean:.6f}")
            print(f"  Benchmark:      {benchmark_mean:.6f}")
            print("=" * 50)

        # Create a DataFrame for the metrics
        metrics_data = {
            'Metric': ['Cumulative Return', 'STDEV of Daily Returns', 'Mean of Daily Returns'],
            'Manual Strategy': [manual_cum_return, manual_std, manual_mean],
            'Benchmark': [benchmark_cum_return, benchmark_std, benchmark_mean]
        }
        metrics_df = pd.DataFrame(metrics_data)

        # Save to CSV in the images folder
        filename = os.path.join(images_dir, f"manual_performance_metrics_{'in_sample' if in_sample else 'out_sample'}.csv")
        metrics_df.to_csv(filename, index=False)
        # print(f"Performance metrics saved to {filename}")

    def process_orders(self, trades, selected_symbol):
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

    def manual_strategy_runner(self, symbol, sd, ed, sv):
        # ms = ManualStrategy.ManualStrategy()
        trades = self.testPolicy(symbol, sd, ed, sv)
        # print("manual strategy ==== trades")
        orders = self.process_orders(trades, symbol)
        value = mktsim.compute_portvals(orders, start_val=sv, commission=9.95, impact=0.005)

        return value

    def benchmark_runner(self, symbol, sd, ed, sv):
        orders = self.benchmarkOrder(symbol, sd, ed, sv)
        value = mktsim.compute_portvals(orders, start_val=sv, commission=0.0, impact=0.0)

        return value

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


        if self.verbose:
            print(orders)
        return orders
    def normalize(self, df):
        return df / df.iloc[0]

    def manual_strategy_in_out_sample_analysis_plotter(self):
        sv = 1000000
        symbol = "JPM"
        insample_sd = dt.datetime(2008, 1, 1)
        insample_ed = dt.datetime(2009, 12, 31)
        outsample_sd = dt.datetime(2010, 1, 1)
        outsample_ed = dt.datetime(2011, 12, 31)

        insample_trades = self.testPolicy(symbol, insample_sd, insample_ed, sv)
        outsample_trades = self.testPolicy(symbol, outsample_sd, outsample_ed, sv)

        insample_benchmark_values = self.benchmark_runner(symbol, insample_sd, insample_ed, sv)
        outsample_benchmark_values = self.benchmark_runner(symbol, outsample_sd, outsample_ed, sv)

        insample_orders = self.process_orders(insample_trades, symbol)
        outsample_orders = self.process_orders(outsample_trades, symbol)

        self.performance_ploter(insample_orders, insample_benchmark_values, True)
        self.performance_ploter(outsample_orders, outsample_benchmark_values, False)

# if __name__ == "__main__":
#     ms = ManualStrategy()
#     ms.manual_strategy_in_out_sample_analysis_plotter()