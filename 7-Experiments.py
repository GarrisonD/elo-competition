# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

torch.__version__

# +
X = torch.tensor([[10, 20, 30],
                  [20, 30, 40],
                  [30, 40, 50],
                  [40, 50, 60],
                  [50, 60, 70]], dtype=torch.float)

y = torch.tensor([[40],
                  [50],
                  [60],
                  [70],
                  [80]], dtype=torch.float)

# MLP (13 epochs)
model = nn.Sequential(nn.Linear(3, 100),
                      nn.ReLU(),
                      nn.Linear(100, 1))

# +
X = torch.tensor([[10, 20, 30],
                  [20, 30, 40],
                  [30, 40, 50],
                  [40, 50, 60],
                  [50, 60, 70]], dtype=torch.float).view(-1, 1, 3)

y = torch.tensor([[40],
                  [50],
                  [60],
                  [70],
                  [80]], dtype=torch.float)

# CNN 1D (10 epochs)
model = nn.Sequential(nn.Conv1d(1, 64, 2),
                      nn.ReLU(),
                      nn.MaxPool1d(2),
                      nn.Flatten(),
                      nn.Linear(64, 50),
                      nn.ReLU(),
                      nn.Linear(50, 1))

# +
X = torch.tensor([[10, 20, 30],
                  [20, 30, 40],
                  [30, 40, 50],
                  [40, 50, 60],
                  [50, 60, 70]], dtype=torch.float).view(-1, 3, 1)

y = torch.tensor([[40],
                  [50],
                  [60],
                  [70],
                  [80]], dtype=torch.float)

# LSTM (20 epochs)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1, 50, batch_first=True)

        self.tail = nn.Sequential(nn.ReLU(), nn.Linear(50, 1))

    def forward(self, x):
        out, _ = self.lstm(x)

        # inverse the activation function applied by CuDNN LSTM
        out = 0.5 * (torch.log(1 + out) / (1 - out))

        return self.tail(out[:, -1])

model = Net()

# +
X = torch.tensor([[10, 20, 30, 40],
                  [20, 30, 40, 50],
                  [30, 40, 50, 60],
                  [40, 50, 60, 70],
                  [50, 60, 70, 80]], dtype=torch.float).view(-1, 2, 2, 1) # .rename('N', 'S', 'L', 'C')

y = torch.tensor([[50],
                  [60],
                  [70],
                  [80],
                  [90]], dtype=torch.float)

# CNN-LSTM (100 epochs)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(nn.Conv1d(1, 64, 1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(2),
                                  nn.Flatten())

        self.lstm = nn.LSTM(64, 50, batch_first=True)

        self.tail = nn.Linear(50, 1)

    def forward(self, x, debug=False):
        N, S, L, C = x.shape
        if debug: print(x.shape)

        x = x.view(N * S, L, C)
        if debug: print(x.shape)

        x = x.rename('N', 'L', 'C')
        x = x.transpose('L', 'C')
        x = x.rename(None)
        if debug: print(x.shape)

        x = self.head(x)
        if debug: print(x.shape)

        x = x.view(N, S, -1)
        if debug: print(x.shape)

        x, _ = self.lstm(x)

        x = x.rename('N', 'S', 'F')
        if debug: print(x.shape)

        x = x.rename(None)
        x = x[:, -1]

        # x = 0.5 * (torch.log(1 + x) / (1 - x))

        return self.tail(x)

model = Net()

# +
X = torch.tensor([[10, 20, 30],
                  [20, 30, 40],
                  [30, 40, 50],
                  [40, 50, 60],
                  [50, 60, 70]], dtype=torch.float).view(-1, 3, 1)

y = torch.tensor([[40, 50],
                  [50, 60],
                  [60, 70],
                  [70, 80],
                  [80, 90]], dtype=torch.float).view(-1, 2, 1)

# LSTM (20 epochs)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.LSTM(  1, 100, batch_first=True)
        self.decoder = nn.LSTM(100, 100, batch_first=True)

        self.tail = nn.Linear(100, 1)

    def forward(self, x, debug=False):
        x, _ = self.encoder(x)
        if debug: print(x.shape)

        x = F.relu(x)
        x = x[:, -1:]
        if debug: print(x.shape)

        x = x.repeat(1, 2, 1)
        if debug: print(x.shape)

        x, _ = self.decoder(x)
        if debug: print(x.shape)

        x = F.relu(x)

        x = self.tail(x)
        if debug: print(x.shape)

        return x

model = Net()
# -

model(X, debug=True)

sum([param.numel() for param in model.parameters() if param.requires_grad])

# +
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X[  :-1], y[  :-1])
valid_dataset = TensorDataset(X[-1:  ], y[-1:  ])

train_loader = DataLoader(train_dataset, shuffle=True)
valid_loader = DataLoader(valid_dataset, shuffle=True)

assert len(train_loader) == 4
assert len(valid_loader) == 1

# +
from torch import optim
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

device = 'cpu'

model = model.to(device)

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters())

n_epochs = 100
valid_loss_min = np.Inf
writer = SummaryWriter('logs/cnn-lstm')

for epoch in range(n_epochs):
    cum_train_loss = 0.
    cum_valid_loss = 0.

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        model.train()

        # ZERO PREVIOUS GRADS
        optimizer.zero_grad()

        y_pred = model.forward(X)
        assert y.shape == y_pred.shape
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        cum_train_loss += loss.item()

    model.eval()

    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model.forward(X)
            assert y.shape == y_pred.shape
            loss = criterion(y_pred, y)

            cum_valid_loss += loss.item()

    train_loss = cum_train_loss / len(train_loader)
    valid_loss = cum_valid_loss / len(valid_loader)

    print(f'Train loss: {train_loss:.5} - Validation loss: {valid_loss:.5} - {y_pred}')

#     if valid_loss < valid_loss_min:
#         print('Validation loss decreased: %.5f => %.5f | Saving model...' % (valid_loss_min, valid_loss))
#         torch.save(model.state_dict(), 'model.pt')
#         valid_loss_min = valid_loss

    writer.add_scalars('loss', dict(train_loss=train_loss, valid_loss=valid_loss), epoch)
