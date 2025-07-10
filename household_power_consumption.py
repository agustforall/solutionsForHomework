# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 采样以提升效率
sampled_train = train.sample(frac=0.1, random_state=42)
sampled_test = test.sample(frac=0.1, random_state=42)

feature_cols = [col for col in train.columns if col not in ['datetime', 'Global_active_power']]
scaler.fit(sampled_train[feature_cols])
train_scaled = scaler.transform(sampled_train[feature_cols])
test_scaled = scaler.transform(sampled_test[feature_cols])

# split X and y
X_train = train_scaled
y_train = sampled_train['Global_active_power'].values
X_test = test_scaled
y_test = sampled_test['Global_active_power'].values

# creat dataloaders
import torch
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# build a LSTM model
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, output_dim)

# train the model
import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb = Xb.unsqueeze(1)  # (batch, seq, feature)
        optimizer.zero_grad()
        pred = model(Xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# evaluate the model on the test set
model.eval()
preds = []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)
        pred = model(Xb).squeeze()
        preds.extend(pred.cpu().numpy())

# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Ground Truth')
plt.plot(preds, label='Predictions')
plt.legend()
plt.show()
