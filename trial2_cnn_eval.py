import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import datautils

BATCH_SIZE = 4

symbols = datautils.get_symbols_list()

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
        self.conv3 = nn.Conv2d(21, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        price, wt = x
        price = price[:, :, 1:, :] # except BTC
        wt = wt[1:]
        wt = wt.view(1, 1, -1, 1)
        output = self.relu(self.conv1(price))
        output = self.relu(self.conv2(output))
        output = torch.cat((wt, output), dim=1)
        output = self.conv3(output)
        output = torch.cat((self.cash_bias, output), dim=2)
        output = output.view(-1, output.shape[2])
        output = self.softmax(output)
        # print(output.shape)
        return output

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        self.c = 0.0025
    
    def forward(self, x, y, last_w):
        prices = y
        changes = price_change(prices)

        wt_prime = (changes * last_w) / torch.sum(changes * last_w, dim=1)
        mu = 1 - (torch.sum(torch.abs(wt_prime - x), dim=1) * self.c)
        portfolio_value = torch.sum(changes * x, dim=1) * mu
        # print(portfolio_value, torch.sum(changes * x, dim=1))

        reward = -torch.mean(torch.log(portfolio_value))
        
        return reward, portfolio_value.squeeze()

def main():
    data_close = datautils.get_global_price()
    # data_low = datautils.get_global_price('low')
    # data_high = datautils.get_global_price('high')

    model = CNN()
    optimizer = optim.Adam(model.parameters())
    criterion = RewardLoss()

    # 35041 - 50 + 1 = 34992
    # Training 0 - 24493 (24494)
    # Validation 24494 - 29742 (5249)
    # Test 29743 - 34991 (5249)
    states = price_state(data_close)
    states = torch.from_numpy(states).float().unsqueeze(1)

    test_ids = range(24494, len(states))
    checkpoint = torch.load('./checkpoints/trial2_cnn.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    wt_old = torch.cat((torch.ones(1), torch.zeros(len(symbols))))
    test_portfolio = []
    test_portfolio.append(wt_old.unsqueeze(0))
    test_pv = []
    with torch.no_grad():
        for i in test_ids:
            model.eval()
            state = states[i].unsqueeze(0)
            input = (state, wt_old)
            output = model(input)
            test_portfolio.append(output)

            loss, portfolio_value = criterion(output, state, wt_old)
            test_pv.append(portfolio_value)
            if i % 1000 == 0:
                print(i, loss)
                print(output)
    
    torch.save({'test_portfolio': test_portfolio}, './checkpoints/trial2_cnn_test.pth.tar')
    test_pv = torch.stack(test_pv).squeeze().detach().numpy()
    plt.plot(np.cumprod(test_pv))
    plt.show()

if __name__ == "__main__":
    main()