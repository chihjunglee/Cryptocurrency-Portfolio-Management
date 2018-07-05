import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from utils import datautils

def main():
    dataset = datautils.get_global_price()

    changes = dataset.copy()
    changes = changes[:, 49:]
    changes[:, 1:] = changes[:, 1:] / changes[:, :-1]
    changes[:, 0] = 1
    print(changes.shape)

    train_changes = changes[:, :24494]
    test_changes = changes[:, 24494:]
    print(test_changes.shape)

    train_checkpoint = torch.load('./checkpoints/trial2_cnn.pth.tar')
    train_portfolio = train_checkpoint['train_portfolio']
    test_checkpoint = torch.load('./checkpoints/trial2_cnn_test.pth.tar')
    test_portfolio = test_checkpoint['test_portfolio']

    train_portfolio = torch.stack(train_portfolio)
    train_portfolio = train_portfolio.squeeze().transpose(0, 1)
    train_portfolio = train_portfolio.detach().numpy()
    print(train_portfolio.shape)
    test_portfolio = torch.stack(test_portfolio)
    test_portfolio = test_portfolio.squeeze().transpose(0, 1)
    test_portfolio = test_portfolio.detach().numpy()
    print(test_portfolio.shape)
    # daily_pv = np.sum(train_changes * train_portfolio, axis=0)
    # daily_pv = np.sum(test_changes * test_portfolio, axis=0)
    # print(test_changes)
    # print(np.prod(daily_pv))
    # print(daily_pv.shape)
    # plt.plot(np.cumprod(daily_pv))
    for i in range(1, 11):
        plt.plot(np.cumprod(test_changes[i]))
    plt.show()

if __name__ == "__main__":
    main()