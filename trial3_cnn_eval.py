import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import datautils
from models.cnn_3ch_2017 import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

symbols = datautils.get_symbols_list()

BATCH_SIZE = 1
NUM_COINS = len(symbols)

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        self.c = 0.0025
    
    def forward(self, x, y, last_w):
        prices = y
        changes = datautils.price_change(prices)

        wt_prime = (changes * last_w) / torch.sum(changes * last_w, dim=1).view(-1, 1)
        mu = 1 - (torch.sum(torch.abs(wt_prime - x)[:, 1:], dim=1) * self.c)
        portfolio_value = torch.sum(changes * last_w, dim=1) * mu

        reward = -torch.mean(torch.log(portfolio_value))
        
        return reward, portfolio_value

def main():
    model = CNN(BATCH_SIZE)
    optimizer = optim.Adam(model.parameters())
    criterion = RewardLoss()

    states = datautils.get_states_tensor(3)

    TEST_START_IDX = 32456 - 49 + 1

    pvm = nn.functional.softmax(torch.ones(states.shape[0] + BATCH_SIZE, NUM_COINS+1), dim=1)
    pvm[TEST_START_IDX] = torch.cat((torch.ones(1), torch.zeros(NUM_COINS)))

    train_ids = range(0, 32456 - BATCH_SIZE + 1)
    test_ids = range(TEST_START_IDX, len(states) + 1 - BATCH_SIZE)
    # test_ids = range(32456, 32457)

    checkpoint = torch.load('./checkpoints/trial3_cnn.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # pvm = checkpoint['pvm']

    wt_old = torch.cat((torch.ones(1), torch.zeros(NUM_COINS))).view(1, -1)
    test_portfolio = []
    # test_weight = []
    # test_weight.append(wt_old.numpy())
    with torch.no_grad():
        for i in test_ids:
            model.eval()
            state = states[i:i+BATCH_SIZE]
            # wt_old = pvm[i:i+BATCH_SIZE]
            input = (state, wt_old)

            output = model(input)
            # pvm[i+1:i+BATCH_SIZE+1] = output.detach()
            # test_weight.append(output.detach().numpy())
            # test_weight.append(datautils.price_change(state).squeeze().numpy())

            loss, pv = criterion(output, state, wt_old)
            test_portfolio.append(pv[-1].detach().numpy())
            if i % 1000 == 0:
                print(i, pv)

            wt_old = output.detach()

    # test_weight = np.array(test_weight)
    # test_weight_df = pd.DataFrame(data=test_weight.squeeze())
    # test_weight_df.to_csv('./data/cnn_changes_3.csv')

    ret = np.array(test_portfolio) - 1 # Returns
    print("Mean =", np.mean(ret), "Std =", np.std(ret))
    print("Sharpe =", np.mean(ret)/np.std(ret))
    print("Final Portfolio Value =", np.prod(test_portfolio))
    # Plot of Portfolio Value over Time 
    plt.plot(np.cumprod(test_portfolio))
    plt.show()

if __name__ == "__main__":
    main()