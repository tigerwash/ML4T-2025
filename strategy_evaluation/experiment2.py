import ManualStrategy
import marketsimcode as mktsim
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import StrategyLearner
import util as ut
import numpy as np


def author():
    return "lliao32"

def study_group():
    return "lliao32"

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

def strategy_learner_runner_with_impact(symbol, sd, ed, sv, impact=0.005):
    sl = StrategyLearner.StrategyLearner(False, impact, 0)
    # in sample training: 2008-01-01 to 2009-12-31
    sl.add_evidence(symbol, dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 1000000)

    trades = sl.testPolicy(symbol, sd, ed, sv)
    # print("strategy learner ==== trades")
    orders = process_orders(trades, symbol)
    value = mktsim.compute_portvals(orders, start_val=sv, commission=0, impact=impact)

    return value

def experiment2_impact_plot(symbol, sd, ed, sv):
    dates = pd.date_range(sd, ed)
    df_container = pd.DataFrame(index=dates)

    impact_a = 0
    impact_b = 0.002
    impact_c = 0.004
    impact_d = 0.006

    sl_result_a = strategy_learner_runner_with_impact(symbol, sd, ed, sv, impact_a)
    sl_result_b = strategy_learner_runner_with_impact(symbol, sd, ed, sv, impact_b)
    sl_result_c = strategy_learner_runner_with_impact(symbol, sd, ed, sv, impact_c)
    sl_result_d = strategy_learner_runner_with_impact(symbol, sd, ed, sv, impact_d)

    df_container[f'Strategy_Learner_Impact_{impact_a}'] = sl_result_a
    df_container[f'Strategy_Learner_Impact_{impact_b}'] = sl_result_b
    df_container[f'Strategy_Learner_Impact_{impact_c}'] = sl_result_c
    df_container[f'Strategy_Learner_Impact_{impact_d}'] = sl_result_d

    plt.figure(figsize=(12, 6))
    plt.plot(df_container[f'Strategy_Learner_Impact_{impact_a}'], label=f'Strategy Learner Impact {impact_a}', color='blue')
    plt.plot(df_container[f'Strategy_Learner_Impact_{impact_b}'], label=f'Strategy Learner Impact {impact_b}', color='green')
    plt.plot(df_container[f'Strategy_Learner_Impact_{impact_c}'], label=f'Strategy Learner Impact {impact_c}', color='orange')
    plt.plot(df_container[f'Strategy_Learner_Impact_{impact_d}'], label=f'Strategy Learner Impact {impact_d}', color='red')
    plt.title('Strategy Learner Performance with Different Impact Values')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend(loc = 'best')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'images/experiment2_impact_line.png')

    return sl_result_a, sl_result_b, sl_result_c, sl_result_d


def experiment2_plot(symbol, sd, ed, sv):
    [sl_result_a, sl_result_b, sl_result_c, sl_result_d] = experiment2_impact_plot(symbol, sd, ed, sv)

    # Calculate final cumulative returns (as single values, not Series)
    cumulative_return_a = sl_result_a.iloc[-1] / sl_result_a.iloc[0] - 1
    cumulative_return_b = sl_result_b.iloc[-1] / sl_result_b.iloc[0] - 1
    cumulative_return_c = sl_result_c.iloc[-1] / sl_result_c.iloc[0] - 1
    cumulative_return_d = sl_result_d.iloc[-1] / sl_result_d.iloc[0] - 1

    # Calculate daily returns for proper Sharpe ratio calculation
    daily_returns_a = sl_result_a.pct_change().dropna()
    daily_returns_b = sl_result_b.pct_change().dropna()
    daily_returns_c = sl_result_c.pct_change().dropna()
    daily_returns_d = sl_result_d.pct_change().dropna()

    # Calculate Sharpe ratios (assuming 252 trading days per year)
    sharpe_ratio_a = (daily_returns_a.mean() / daily_returns_a.std()) * np.sqrt(252)
    sharpe_ratio_b = (daily_returns_b.mean() / daily_returns_b.std()) * np.sqrt(252)
    sharpe_ratio_c = (daily_returns_c.mean() / daily_returns_c.std()) * np.sqrt(252)
    sharpe_ratio_d = (daily_returns_d.mean() / daily_returns_d.std()) * np.sqrt(252)

    # Prepare data for bar charts
    impact_values = ['0', '0.002', '0.004', '0.006']
    cumulative_returns = [cumulative_return_a, cumulative_return_b, cumulative_return_c, cumulative_return_d]
    sharpe_ratios = [sharpe_ratio_a, sharpe_ratio_b, sharpe_ratio_c, sharpe_ratio_d]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot cumulative returns
    bar_width = 0.35
    x_pos = np.arange(len(impact_values))
    ax1.bar(x_pos, cumulative_returns, bar_width, color='blue', alpha=0.7)
    ax1.set_xlabel('Impact Value')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns by Impact Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(impact_values)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Add value labels on top of bars
    for i, v in enumerate(cumulative_returns):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # Plot Sharpe ratios
    ax2.bar(x_pos, sharpe_ratios, bar_width, color='green', alpha=0.7)
    ax2.set_xlabel('Impact Value')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio by Impact Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(impact_values)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Add value labels on top of bars
    for i, v in enumerate(sharpe_ratios):
        ax2.text(i, v + 0.1, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('images/experiment2_metrics_bar.png')
    plt.close()
