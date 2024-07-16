import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_binary_X_Y(df):
    X = []
    Y = []
    for row_id in range(len(df) - 1):
        for stock_id in range(1, 51):
            stock_columns = [col for col in df.columns if col.startswith(f'Stock_{stock_id}_')]
            stock_columns.append(f'Stock_{stock_id}')
            x = []
            for column_name in stock_columns:
                x.append(df.iloc[row_id][column_name])
            
            X.append(x)
            # Convert to binary classification: 1 if price increased, 0 if decreased or stayed the same
            price_change = df.iloc[row_id + 1][f'Stock_{stock_id}'] - df.iloc[row_id][f'Stock_{stock_id}']
            Y.append(1 if price_change > 0 else 0)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64) 
    return X, Y



# Data preparation
df = pd.read_csv('stock_data_with_indicators.csv')
training_org_df = df[49:375]
testing_org_df = df[375:]

X_train, Y_train = create_binary_X_Y(training_org_df)
X_test, Y_test = create_binary_X_Y(testing_org_df)

# Normalize the input data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

X_train = torch.from_numpy(X_train).float().to(device)
X_val = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(Y_train).long().to(device)
y_val = torch.from_numpy(Y_test).long().to(device)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize model
input_size = X_train.shape[1]
model = SimpleClassifier(input_size).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Evaluation
model.eval()
with torch.no_grad():
    test_sequence = X_val[:3]
    outputs = model(test_sequence)
    probabilities = torch.softmax(outputs, dim=1)
    _, predictions = torch.max(probabilities, 1)
    print(f'Probabilities: {probabilities}')
    print(f'Predictions: {predictions}')
    print(f'Actual labels: {y_val[:3]}')