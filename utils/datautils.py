import numpy as np
import pandas as pd
import os
import torch

def get_symbols_list():
    # symbols = ['dash', 'dcr', 'eth', 'ltc', 'sc', 'str', 'xmr', 'xrp']
    # symbols = ['eth', 'ltc', 'xrp', 'etc', 'dash', 'xmr', 'xem', 'fct', 'gnt', 'zec'] # used in paper
    symbols = ['eth', 'ltc', 'xrp', 'etc', 'dash', 'xmr', 'xem', 'fct', 'gnt', 'usdt', 'zec']
    return symbols

def get_global_price(column='close'):
    """
    Construct Global Price Matrix
    Read all json data, then take the closing price
    Insert BTC as 1 at the top row
    """
    eth = pd.read_json('./data/json/btc_eth.json')
    eth.set_index('date')
    times = pd.DataFrame(index=eth['date'])

    prices = []
    for sym in get_symbols_list():
        coin = pd.read_json('./data/json/btc_{}.json'.format(sym))
        coin = coin.set_index('date')
        coin_join = times.join(coin[column])
        coin_join = coin_join.fillna(method='bfill')
        prices.append(coin_join[column])
    prices = np.array(prices) # n x t shaped (n = num of assets, t = num of period)
    prices = np.insert(prices, 0, 1, axis=0) # (n+1) x t (btc on the top)
    return prices

def get_global_price_paper(column='close', backtest=1):
    # Total 34913
    # Train 32457 (0-32456)
    # Test 2457 (32456-34912)
    times = pd.DataFrame(index=get_timeframe_paper(backtest=backtest))

    prices = []
    for sym in get_symbols_list():
        coin = pd.read_json('./data/json/paper_btc_{}.json'.format(sym))
        coin = coin.set_index('date')
        coin_join = times.join(coin[column])
        coin_join = coin_join.fillna(method='bfill').fillna(1)
        prices.append(coin_join[column])
    prices = np.array(prices) # n x t shaped (n = num of assets, t = num of period)
    prices = np.insert(prices, 0, 1, axis=0) # (n+1) x t (btc on the top)
    return prices

def get_timeframe_paper(backtest=1):
    if backtest == 1:
        return pd.date_range('2014-11-01 00:00', '2016-10-28 08:00', freq='30min')
    elif backtest == 2:
        return pd.date_range('2015-02-01 00:00', '2017-01-27 08:00', freq='30min')
    else:
        return pd.date_range('2015-05-01 00:00', '2017-04-27 08:00', freq='30min')

def price_state(prices, period_window=50):
    num_of_assets = prices.shape[0] # n
    num_of_periods = prices.shape[1] # t
    states = np.zeros((num_of_periods - period_window + 1, num_of_assets, period_window)) # t - period window x n x period_window
    for i in range(period_window, num_of_periods + 1):
        states[i - period_window] = normalize_prices(prices[:, i - period_window:i])
    return states

def price_change(prices):
    """
    Construct Price Change Matrix
    Element-wise divison of price at (t+1) with price at (t)
    """
    # changes = prices.copy()
    # changes[:, 1:] = changes[:, 1:] / changes[:, :-1]
    # return changes[:, -1]
    changes = prices.clone()
    changes = changes[:, -1:, :, :]
    changes[:, :, :, 1:] = changes[:, :, :, 1:] / changes[:, :, :, :-1]
    return changes[:, :, :, -1].view(-1, len(get_symbols_list())+1)

def normalize_prices(prices):
    output = prices.copy()
    output[:] = output[:] / output[:, -1].reshape(-1, 1)
    return output

def get_states_tensor(backtest=1):
    data_close = get_global_price_paper('close', backtest=backtest)
    data_low = get_global_price_paper('low', backtest=backtest)
    data_high = get_global_price_paper('high', backtest=backtest)

    states_close = torch.from_numpy(price_state(data_close)).float()
    states_low = torch.from_numpy(price_state(data_low)).float()
    states_high = torch.from_numpy(price_state(data_high)).float()
    states = torch.stack((states_low, states_high, states_close), dim=1)

    return states

def get_daily_return(prices):
    """
    Construct Price Change Matrix
    Element-wise divison of price at (t+1) with price at (t)
    """
    changes = prices.copy()
    changes[:, 1:] = changes[:, 1:] / changes[:, :-1] - 1
    changes[:, 0] = 0
    return changes