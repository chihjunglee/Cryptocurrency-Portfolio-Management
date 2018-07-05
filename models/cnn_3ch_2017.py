import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, batch_size=1):
        super(CNN, self).__init__()
        self.batch_size = batch_size

        self.cash_bias = nn.Parameter(torch.ones(self.batch_size, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 1, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(1, 20, kernel_size=(1, 48))
        self.conv3 = nn.Conv2d(21, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        price, wt = x
        price = price[:, :, 1:, :] # except BTC
        wt = wt[:, 1:].view(self.batch_size, 1, -1, 1)
        output = self.relu(self.conv1(price))
        output = self.relu(self.conv2(output))
        output = torch.cat((wt, output), dim=1)
        output = self.conv3(output)
        output = torch.cat((self.cash_bias, output), dim=2)
        output = output.view(-1, output.shape[2])
        output = self.softmax(output)
        return output