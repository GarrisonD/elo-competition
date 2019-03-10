# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %run ../0-utils/0-Base.py

# Define a dataset:

# +
from torch.utils.data import Dataset

class EloDataset(Dataset):
    def __init__(self, X, y=None):
        if y is not None:
            assert X.shape[0] == y.shape[0]
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
        else:
            self.X = X.astype(np.float32)
            self.y = None
        
    def __len__(self): return self.X.shape[0]
    
    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        else:
            return self.X[index]


# -


# Define *train* and *test* data loaders:

# +
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

X = np.load("../data/3-preprocessed/train/X.npy")
y = np.load("../data/3-preprocessed/train/y.npy")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=13)

train_dataset = EloDataset(X_train, y_train)
valid_dataset = EloDataset(X_valid, y_valid)

kwargs = dict(shuffle=True, batch_size=32)

train_loader = DataLoader(train_dataset, **kwargs)
valid_loader = DataLoader(valid_dataset, **kwargs)

sns.distplot(y_train)
sns.distplot(y_valid)
# -

# Define a device that will be used for training / evaluation:

# +
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# +
from torch import nn

import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=26,
                            hidden_size=64,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

# +
from torch import optim
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter

model = Regressor().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

n_epochs = 50
valid_loss_min = np.Inf
writer = SummaryWriter("runs/test-6")

for epoch in tqdm(range(n_epochs)):
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

    train_loss = (cum_train_loss / len(train_loader)) ** 0.5
    valid_loss = (cum_valid_loss / len(valid_loader)) ** 0.5

    if valid_loss < valid_loss_min:
        print("Validation loss decreased: %.5f => %.5f | Saving model..." % (valid_loss_min, valid_loss))
        torch.save(model.state_dict(), "model.pt")
        valid_loss_min = valid_loss

    writer.add_scalars("loss", dict(train_loss=train_loss, valid_loss=valid_loss), epoch)

# +
# X = np.load("../data/4-features-combined/test/X.npy")

# test_dataset = EloDataset(X)

# test_loader = DataLoader(test_dataset, batch_size=2048)

# model = Regressor().to(device).eval()
# model.load_state_dict(torch.load("model-3.7665.pt"))

# y_test = []

# with torch.no_grad():
#     for x in tqdm(test_loader):
#         x = x.to(device)

#         y_pred = model.forward(x)
#         y_test.append(y_pred.cpu().numpy())

# +
# y_test = np.concatenate(y_test); y_test

# +
# submission_df = pd.read_csv("../data/raw/sample_submission.csv")
# submission_df.target = y_test
# submission_df

# +
# submission_df.to_csv("../submission.csv", index=False)
