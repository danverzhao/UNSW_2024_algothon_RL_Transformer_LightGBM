import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import os
import sys
from collections import Counter
import joblib
#import matplotlib.pyplot as plt

# print('executable: ', sys.executable)
# print('sys.version: ', sys.version)
# print('os.envirom: ', os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment'))
# import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sequence_length = 50

def create_binary_X_Y(df, days_in_future):
    X = []
    Y = []
    for row_id in range(len(df) - (sequence_length + days_in_future)):
        for stock_id in range(1, 51):
            stock_columns = [col for col in df.columns if col.startswith(f'Stock_{stock_id}_')]
            for stock_id_tmp in range(1, 51):
                stock_columns.append(f'Stock_{stock_id_tmp}')
            X_mid = []
            for seq_id in range(sequence_length):  
                x_mid = []
                for column_name in stock_columns:
                    x_mid.append(df.iloc[row_id + seq_id][column_name])
                x_mid.append(stock_id)
                X_mid.append(x_mid)
            X.append(X_mid)

            price_change = df.iloc[row_id + sequence_length + days_in_future][f'Stock_{stock_id}'] - df.iloc[row_id + sequence_length - 1][f'Stock_{stock_id}']
            Y.append([1 if price_change > 0 else 0])
    
    X = np.array(X)
    Y = np.array(Y) 
    return X, Y


def save_binary_data(X, Y, filename_prefix):
    np.save(f"{filename_prefix}_X.npy", X)
    np.save(f"{filename_prefix}_Y.npy", Y)
    print(f"Data saved as {filename_prefix}_X.npy and {filename_prefix}_Y.npy")

def load_binary_data(filename_prefix):
    X = np.load(f"{filename_prefix}_X.npy")
    Y = np.load(f"{filename_prefix}_Y.npy")
    print(f"Data loaded from {filename_prefix}_X.npy and {filename_prefix}_Y.npy")
    return X, Y

# Check if processed data already exists
if os.path.exists("train_3daysinfuture_X.npy") and os.path.exists("test_3daysinfuture_Y.npy"):
    X_train, Y_train = load_binary_data("train_3daysinfuture")
    X_test, Y_test = load_binary_data("test_3daysinfuture")

else:
    print('no saved data, creating new data...')
    df = pd.read_csv('stock_data_with_indicators.csv')
    training_org_df = df[21:375]
    testing_org_df = df[375:]

    X_train, Y_train = create_binary_X_Y(training_org_df, days_in_future=7)
    X_test, Y_test = create_binary_X_Y(testing_org_df, days_in_future=7)
    
    print('data created')
    # Save the processed data
    save_binary_data(X_train, Y_train, "train_7daysinfuture")
    save_binary_data(X_test, Y_test, "test_7daysinfuture")


print(f'X_train.shape: {X_train.shape}')

# for row in range(len(X_train[0])):
#     print(X_train[0][row])

# Normalize the input data
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)


# Save the scaler
if os.path.exists('scaler_transformer.save'):
    pass
else:
    joblib.dump(scaler, 'scaler_transformer.save')
    print("Scaler saved as 'scaler.save'")


print(f'X_train.shape: {X_train_scaled.shape}')
print(f'Y_train.shape: {Y_train.shape}')
print(f'X_test.shape: {X_test_scaled.shape}')
print(f'Y_test.shape: {Y_test.shape}')

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_fc = nn.Linear(input_dim, d_model)
        self.batch_norm1 = nn.BatchNorm1d(d_model)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, src):
        # Residual connection
        residual = self.residual_fc(src)

        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        # Add residual connection
        output = output + residual

        output = output.mean(dim=1)  # Global average pooling
        output = self.batch_norm1(output)
        output = self.dropout(self.relu(self.fc1(output)))
        output = self.batch_norm2(output)
        output = self.fc2(output)
        return torch.sigmoid(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)



# Hyperparameters
batch_size = 128
learning_rate = 0.005
num_epochs = 300
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 128
validation_split = 0.2

print(X_train.shape)
print(Y_train.shape)

# Prepare data
X_train = torch.FloatTensor(X_train_scaled).to(device)
Y_train = torch.FloatTensor(Y_train).to(device) #squeeze(0).to(device)
X_test = torch.FloatTensor(X_test_scaled).to(device)
Y_test = torch.FloatTensor(Y_test).to(device) #squeeze(0).to(device)

# testing tmp no normalisation
# X_train = torch.FloatTensor(X_train).to(device)
# Y_train = torch.FloatTensor(Y_train).to(device) #squeeze(0).to(device)
# X_test = torch.FloatTensor(X_test).to(device)
# Y_test = torch.FloatTensor(Y_test).to(device) #squeeze(0).to(device)




print(X_train.shape)
print(Y_train.shape)

dataset = TensorDataset(X_train, Y_train)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

input_dim = X_train.shape[2]
model = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)

criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming your model is called 'loaded_transformer'
total_params = count_parameters(model)
print(f"The model has {total_params:,} trainable parameters")


def calculate_accuracy(outputs, targets):
    predicted = (outputs > 0.5).float()
    correct = (predicted == targets).float().sum()
    return correct / len(targets)

def calculate_class_balance(predictions, targets):
    predicted_classes = (predictions > 0.5).float().cpu().numpy()
    predicted_counter = Counter(predicted_classes.flatten().tolist())
    actual_counter = Counter(targets.cpu().numpy().flatten().tolist())
    zeros_count_true = 0
    ones_count_true = 0
    zeros_count_pred = 0
    ones_count_pred = 0
    num_ones_true = torch.sum(torch.tensor(targets)).item()
    num_zeros_true = targets.numel() - num_ones_true
    zeros_count_true += num_zeros_true
    ones_count_true += num_ones_true

    num_ones_pred = torch.sum(torch.tensor(predicted_classes)).item()
    num_zeros_pred = torch.tensor(predicted_classes).numel() - num_ones_pred
    zeros_count_pred += num_zeros_pred
    ones_count_pred += num_ones_pred

    # print(f'pred ones ratio: {ones_count_pred}/{ones_count_pred + zeros_count_pred}')
    # print(f'true ones ratio: {ones_count_true}/{ones_count_true + zeros_count_true}')
    return predicted_counter, actual_counter



train_losses = []
val_losses = []

# early stopping parameters
patience = 40
best_val_loss = float('inf')
counter = 0


# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        #batch_y = batch_y.unsqueeze(1)
        #outputs = outputs.squeeze()

        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        total_train_acc += calculate_accuracy(outputs, batch_y)
    
    model.eval()
    total_val_loss = 0
    all_val_outputs = []
    all_val_targets = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            total_val_loss += val_loss.item()
            all_val_outputs.append(outputs)
            all_val_targets.append(batch_y)
            
    
    
    all_val_outputs = torch.cat(all_val_outputs)
    all_val_targets = torch.cat(all_val_targets)
    predicted_balance, actual_balance = calculate_class_balance(all_val_outputs, all_val_targets)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    avg_train_acc = total_train_acc / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Predicted class balance: {dict(predicted_balance)}, Actual class balance: {dict(actual_balance)}')
    
    # Early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    
        torch.save(model.state_dict(), f'3days_best_model_{best_val_loss:.4f}.pth')
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    


# After the training loop, plot the losses
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

