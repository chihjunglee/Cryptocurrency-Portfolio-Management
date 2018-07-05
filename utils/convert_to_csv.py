import numpy as np
import pandas as pd

def get_symbols_list():
    symbols = ['eth', 'ltc', 'xrp', 'etc', 'dash', 'xmr', 'xem', 'fct', 'gnt', 'usdt', 'zec']
    return symbols

def get_timeframe(backtest=1):
    if backtest == 1:
        return pd.date_range('2016-09-07 04:00', '2016-10-28 08:00', freq='30min')
    elif backtest == 2:
        return pd.date_range('2016-12-08 04:00', '2017-01-27 08:00', freq='30min')
    else:
        return pd.date_range('2017-03-07 04:00', '2017-04-27 08:00', freq='30min')

def get_timeframe_train(backtest=1):
    if backtest == 1:
        return pd.date_range('2014-11-01 00:00', '2016-09-07 04:00', freq='30min')
    elif backtest == 2:
        return pd.date_range('2015-02-01 00:00', '2016-12-08 04:00', freq='30min')
    else:
        return pd.date_range('2015-05-01 00:00', '2017-03-07 04:00', freq='30min')

def read_all_jsons(backtest=1):
    # times = pd.DataFrame(index=get_timeframe(backtest=backtest))
    times = pd.DataFrame(index=get_timeframe_train(backtest=backtest))

    for sym in get_symbols_list():
        coin = pd.read_json('./data/json/paper_btc_{}.json'.format(sym))
        coin = coin.set_index('date')
        times = times.join(coin['close'])
        times = times.rename(index=str, columns={'close': str.upper(sym)})
    
    return times

def calculate_returns(df):
    returns = df.copy()
    returns[1:] = (df[1:] / df[:-1].values) -1
    returns = returns.iloc[1:]
    returns = returns.fillna(0)
    return returns

def save_to_csv(df, filename='prices'):
    df.to_csv('./data/{}.csv'.format(filename))

if __name__ == "__main__":
    prices = read_all_jsons(2)
    returns = calculate_returns(prices)
    # save_to_csv(returns, 'returns_train_2')
    # save_to_csv(prices, 'prices_1')