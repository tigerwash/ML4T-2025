from util import get_data, plot_data
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def author():
    return "lliao32"

def study_group():
    return "lliao32"

# Indicator 1: MACD (Moving Average Convergence Divergence) indicator
def macd(prices, fast_period=12, slow_period=26, signal_period=9):
    macd = prices.ewm(span=fast_period, min_periods=fast_period, adjust=False).mean() - prices.ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
    return macd

def macd_signal(macd, signal_period=9):
    signal = macd.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    return signal

def macd_hist(macd, signal):
    hist = macd - signal
    return hist

# Indicator 2: Bollinger Bands Percent
def bbp(prices, window=20):
    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    bbp = (prices - lower_band) / (upper_band - lower_band)
    return bbp

# Indicator 3: Relative Strength Index
def rsi(prices, window=14):
    daily_rets = prices.pct_change()
    up_rets = daily_rets.where(lambda n: n>0, other = 0.0)
    down_rets = -daily_rets.where(lambda n: n<0, other = 0.0)
    up_gain = up_rets.rolling(window=window, min_periods=window).mean()
    down_loss = down_rets.rolling(window=window, min_periods=window).mean()
    rs = up_gain / down_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

# Indicator 4: Momentum
def momentum(prices, window=14):
    momentum = prices / prices.shift(window) - 1
    return momentum

# Indicator 5: Stochastic Oscillator
def stochastic_oscillator(prices, window=14):
    min = prices.rolling(window=window, min_periods=window).min()
    max = prices.rolling(window=window, min_periods=window).max()
    stoch_k = (prices - min) / (max - min)
    return stoch_k


def MACD_plot(prices):
    macd_val = macd(prices)
    signal_val = macd_signal(macd_val)
    hist_val = macd_hist(macd_val, signal_val)

    plt.figure(figsize=(10, 5))
    # plt.plot(prices, label='Price')
    plt.plot(macd_val, label='MACD')
    plt.plot(signal_val, label='Signal Line')
    plt.title('Macd Indicator')

    colors = np.where(hist_val >= 0, 'green', 'red')
    plt.bar(hist_val.index, hist_val, color=colors, label='MACD Histogram')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()
    plt.title('MACD Indicator')
    # plt.show()
    plt.savefig('MACD_plot.png')

def BBP_plot(prices):
    bbp_val = bbp(prices)
    plt.figure(figsize=(10, 5))
    plt.plot(bbp_val, label='BBP')

    plt.axhline(y=1.0, color='green', linestyle='dashed', linewidth=1, label='Overbought')
    plt.axhline(y=0.0, color='red', linestyle='dashed', linewidth=1, label='Oversold')
    plt.title('Bollinger Bands Percent')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('BBP_plot.png')

def RSI_plot(prices):
    rsi_val = rsi(prices)
    plt.figure(figsize=(10, 5))
    plt.plot(rsi_val, label='RSI')

    plt.axhline(y=70, color='green', linestyle='dashed', linewidth=1, label='Overbought')
    plt.axhline(y=30, color='red', linestyle='dashed', linewidth=1, label='Oversold')
    plt.title('Relative Strength Index')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('RSI_plot.png')

def MOM_plot(prices):
    mom_val = momentum(prices)
    plt.figure(figsize=(10, 5))

    # Plot Momentum Line
    plt.plot(mom_val, label='Momentum', color='grey', linewidth=1)

    plt.fill_between(mom_val.index, mom_val, where=(mom_val > 0), color='green', alpha=0.3, label='Bullish Momentum')
    plt.fill_between(mom_val.index, mom_val, where=(mom_val < 0), color='red', alpha=0.3, label='Bearish Momentum')

    # Add horizontal zero line
    plt.axhline(0, linestyle='dashed', color='black', linewidth=1)


    plt.title('Momentum Indicator')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('MOM_plot.png')

def stoch_plot(prices):
    stoch_val = stochastic_oscillator(prices)
    plt.figure(figsize=(10, 5))
    plt.plot(stoch_val, label='Stochastic Oscillator')

    plt.axhline(y=0.8, color='green', linestyle='dashed', linewidth=1, label='Overbought')
    plt.axhline(y=0.2, color='red', linestyle='dashed', linewidth=1, label='Oversold')
    plt.title('Stochastic Oscillator')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('Stoch_plot.png')

def JPM_prices_plot(prices):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label='JPM Prices')
    plt.title('JPM Stock Prices')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # plt.show()
    plt.savefig('JPM_prices_plot.png')

def run():

    # benchmark_val = benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    JPM_Prices = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)), False)
    JPM_Prices = JPM_Prices.ffill().bfill()
    # print(JPM_Prices)
    JPM_prices_plot(JPM_Prices['JPM'])
    MACD_plot(JPM_Prices['JPM'])
    BBP_plot(JPM_Prices['JPM'])
    RSI_plot(JPM_Prices['JPM'])
    MOM_plot(JPM_Prices['JPM'])
    stoch_plot(JPM_Prices['JPM'])


def sma(param):
    return None