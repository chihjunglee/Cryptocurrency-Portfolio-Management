import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

symbols = ['dash', 'dcr', 'eth', 'ltc', 'sc', 'str', 'xmr', 'xrp']
# symbols = ['eth', 'ltc', 'xrp', 'etc', 'dash', 'xmr', 'xem', 'fct', 'gnt', 'zec']

def global_price():
    """
    Construct Global Price Matrix
    Read all json data, then take the closing price
    Insert BTC as 1 at the top row
    """
    prices = []
    for sym in symbols:
        coin = pd.read_json('./data/json/btc_{}.json'.format(sym))
        prices.append(coin.close)
    prices = np.array(prices) # n x t shaped (n = num of assets, t = num of period)
    prices = np.insert(prices, 0, 1, axis=0) # (n+1) x t (btc on the top)
    return prices

def price_state(prices, period_window=50):
    num_of_assets = prices.shape[0] # n
    num_of_periods = prices.shape[1] # t
    states = np.zeros((num_of_periods - period_window, num_of_assets, period_window)) # t - period window x n x period_window
    for i in range(period_window, num_of_periods):
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
    # print(changes.shape)
    changes[:, :, :, 1:] = changes[:, :, :, 1:] / changes[:, :, :, :-1]
    return changes[:, :, :, -1].view(-1, len(symbols)+1)

def normalize_prices(prices):
    output = prices.copy()
    output[:] = output[:] / output[:, -1].reshape(-1, 1)
    return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cash_bias = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(1, 20, kernel_size=(1, 48))
        self.conv3 = nn.Conv2d(20, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        price, wt = x
        price = price[:, :, 1:, :] # except BTC
        wt = wt[1:]
        wt = wt.view(1, 1, -1, 1)
        output = self.relu(self.conv1(price))
        output = self.relu(self.conv2(output))
        # output = torch.cat((wt, output), dim=1)
        output = self.conv3(output)
        output = torch.cat((self.cash_bias, output), dim=2)
        output = output.view(-1, output.shape[2])
        output = self.softmax(output)
        # print(output.shape)
        return output

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
    
    def forward(self, x, y):
        prices, accum_loss = y
        changes = price_change(prices)
        # print(x.shape, changes.shape)
        reward = -torch.sum(x * changes)
        # reward = -(torch.sum(x * changes) * (-accum_loss))
        # print(reward)
        # reward = -(x.squeeze().dot(changes.squeeze()) * -accum_loss)
        # reward = -(x.squeeze().dot(changes.squeeze()))
        
        return reward

def main():
    model = CNN()
    optimizer = optim.Adam(model.parameters())
    criterion = RewardLoss()

    # 35041 - 50 + 1 = 34992
    # Training 0 - 24493 (24494)
    # Validation 24494 - 29742 (5249)
    # Test 29743 - 34991 (5249)
    dataset = global_price()
    states = price_state(dataset)
    states = torch.from_numpy(states).float().unsqueeze(1)

    train_ids = range(18000, 24494)
    test_ids = range(24494, len(states))

    wt_old = torch.from_numpy(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])).float()
    accum_loss = torch.tensor([1.])

    accum_rewards_train = []
    for i in train_ids:
    # for i in range(5):
        model.train()
        # prices = dataset[:, i:i+50]
        # prices = normalize_prices(prices)
        # prices = torch.from_numpy(prices).float().unsqueeze(0).unsqueeze(0)
        # print(prices.shape)
        state = states[i].unsqueeze(0)

        input = (state, wt_old)
        output = model(input)
        # print(output)
        wt_old = output.squeeze()

        loss = criterion(output, (state, accum_loss))
        # accum_loss = accum_loss * -loss
        # accum_loss = loss
        accum_rewards_train.append(-loss)
        if i % 1000 == 0:
            print(i, loss)
            print(output)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # for param in model.parameters():
        #     print(param.data)
    
    accum_rewards_train = torch.tensor(accum_rewards_train)
    print("Cummulative Return Train", torch.prod(accum_rewards_train))

    wt_old = torch.from_numpy(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.])).float()
    # accum_loss = torch.tensor([1.])
    accum_rewards = []
    with torch.no_grad():
        for i in test_ids:
            model.eval()
            state = states[i].unsqueeze(0)
            input = (state, wt_old)
            output = model(input)

            loss = criterion(output, (state, accum_loss))
            accum_rewards.append(-loss)
            if i % 1000 == 0:
                print(i, loss)
                print(output)

    accum_rewards = torch.tensor(accum_rewards)
    print("Cummulative Return Test", torch.prod(accum_rewards))

if __name__ == "__main__":
    # price_change()
    main()
