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
TRAIN_EPOCH = 3
LEARNING_RATE = 3e-4
NUM_COINS = len(symbols)

class RewardLoss(nn.Module):
    def __init__(self):
        super(RewardLoss, self).__init__()
        self.c = 0.0025
    
    def forward(self, x, y, last_w, future_changes):
        prices = y
        changes = datautils.price_change(prices)

        wt_prime = (changes * last_w) / torch.sum(changes * last_w, dim=1).view(-1, 1)
        mu = 1 - (torch.sum(torch.abs(wt_prime - x), dim=1) * self.c)
        # portfolio_value = torch.sum(future_changes * x, dim=1) * mu
        portfolio_value = torch.sum(changes * last_w, dim=1) * mu
        # print(portfolio_value)

        reward = -torch.mean(torch.log(portfolio_value))
        
        return reward, portfolio_value

def main():
    model = CNN(BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = RewardLoss()

    # 34913 - 50 + 1 = 34864
    # Train 32457 (0-32456)
    # Test 2457 (32456-34912)
    states = datautils.get_states_tensor(3)

    pvm = nn.functional.softmax(torch.randn(states.shape[0], NUM_COINS+1), dim=1)
    pvm[0] = torch.cat((torch.ones(1), torch.zeros(NUM_COINS)))

    train_ids = range(0, 32456 - BATCH_SIZE + 1)
    # test_ids = range(32456, len(states) - BATCH_SIZE + 1)

    train_loss = 0
    train_step = 2000
    train_pv = []
    for e in range(TRAIN_EPOCH):
        for i in train_ids:
            model.train()
            state = states[i:i+BATCH_SIZE]
            future_changes = datautils.price_change(states[i+1:i+BATCH_SIZE+1])
            wt_old = pvm[i:i+BATCH_SIZE]
            input = (state, wt_old)

            output = model(input)
            pvm[i+1:i+BATCH_SIZE+1] = output

            loss, pv = criterion(output, state, wt_old, future_changes)
            train_loss += loss.item()
            train_pv.append(pv.detach().numpy())

            if i % train_step == 0:
                print(i, "Loss =", train_loss / train_step)
                print(output)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print("Train PV =", np.prod(train_pv))
        train_pv = []
        # Save Checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pvm': pvm,
        }
        torch.save(checkpoint, './checkpoints/trial3_cnn.pth.tar')

if __name__ == "__main__":
    main()