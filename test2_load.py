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



def get_features_overtime(df_prices, seq_length):
    df_prices = df_prices.T #(250, 50)
    df_prices = pd.DataFrame(df_prices)
    #print(f'1st row: {df_prices.iloc[0][0]}')
    #print(f'last row: {df_prices.iloc[-1][0]}') # last row is the newest
    final_list = []

    for stock in df_prices.columns:
    
        SMA20 = df_prices[stock].rolling(window=20).mean().iloc[-seq_length:]
        SMA50 = df_prices[stock].rolling(window=50).mean().iloc[-seq_length:]
        SMA10 = df_prices[stock].rolling(window=10).mean().iloc[-seq_length:]
       
        # Exponential Moving Average (EMA) - 10/20/50 days
        EMA20 = df_prices[:][stock].ewm(span=20, adjust=False).mean().iloc[-seq_length:]
        EMA50 = df_prices[:][stock].ewm(span=50, adjust=False).mean().iloc[-seq_length:]
        EMA10 = df_prices[:][stock].ewm(span=10, adjust=False).mean().iloc[-seq_length:]
        
        # Relative Strength Index (RSI) - 14 days
        delta = df_prices[:][stock].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-seq_length:]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-seq_length:]
        rs = gain / loss
        RSI14 = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - 20 days, 2 standard deviations
        BB_middle = df_prices[:][stock].rolling(window=20).mean().iloc[-seq_length:]
        BB_upper = BB_middle + 2 * df_prices[:][stock].rolling(window=20).std().iloc[-seq_length:]
        BB_lower = BB_middle - 2 * df_prices[:][stock].rolling(window=20).std().iloc[-seq_length:]
        
        # Moving Average Convergence Divergence (MACD)
        exp1 = df_prices[:][stock].ewm(span=12, adjust=False).mean()
        exp2 = df_prices[:][stock].ewm(span=26, adjust=False).mean()
        MACD = exp1 - exp2
        MACD_signal = MACD.ewm(span=9, adjust=False).mean().iloc[-seq_length:]
        MACD = (exp1 - exp2).iloc[-seq_length:]
        
        all_price = df_prices.iloc[-seq_length:][:]
        
        # print(f'SMA: {SMA10} {SMA20} {SMA50}')
        # print(f'EMA: {EMA10} {EMA20} {EMA50}')
        # print(f'RSI: {RSI14}')
        # print(f'BB: {BB_lower} {BB_middle}, {BB_upper}')
        # print(f'MACD: {MACD} {MACD_signal}')
        
        custom_input = [SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal]

        column_series_list = [all_price[column] for column in all_price.columns]

       
        for stock_prices in column_series_list:
            custom_input.append(stock_prices)

        series_of_stock_id = pd.Series([stock+1] * 50)
        custom_input.append(series_of_stock_id)

        final_list.append(np.array(custom_input))

    return np.array(final_list)


# Hyperparameters
batch_size = 128
learning_rate = 0.005
num_epochs = 300
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 128
validation_split = 0.2
input_dim = 63

model = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward)

# Load the saved model state
model_file = '3days_best_model_0.5724.pth'
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
model.eval()


# # Load the saved scaler
# scaler = joblib.load('scaler_transformer.save')


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices750.txt"
prcAll = loadPrices(pricesFile)
print(f'prcAll shape: {prcAll.shape}')
print("Loaded %d instruments for %d days" % (nInst, nt))


df_price = prcAll[:, :490]

def whole_data_prepare(df_price):

    inputs = get_features_overtime(df_price, seq_length=50)

    #print(f'{inputs.shape[0]} stocks, {inputs.shape[1]} features, {inputs.shape[2]} sequence length')

    inputs = np.transpose(inputs, (0, 2, 1))


    # Prepare your input data
    def normalise_input_data(raw_data):
        # Assuming raw_data is a numpy array of shape (sequence_length, input_dim)
        # making a new scaler
        scaler = StandardScaler()
        raw_data_reshaped = raw_data.reshape(-1, raw_data.shape[-1])
        raw_data_scaled = scaler.fit_transform(raw_data_reshaped).reshape(raw_data.shape)

        # having the loaded scaler 
        # raw_data_reshaped = raw_data.reshape(-1, raw_data.shape[-1])
        # raw_data_scaled = scaler.transform(raw_data_reshaped).reshape(raw_data.shape)


        return torch.FloatTensor(raw_data_scaled)  # Add batch dimension

    scaled_inputs = normalise_input_data(inputs)

    return scaled_inputs


# Perform inference
def predict(model, input_data):
    with torch.no_grad():
        output = model(input_data)
    return output

# sample usage for prediction
scaled_inputs = whole_data_prepare(df_price)
prediction = predict(model, scaled_inputs)
# print(f"Prediction first 5:\n {prediction[:5]}")


transformed_prediction = (prediction > 0.5).float()
# print(f"Transformed prediction:\n {transformed_prediction[:5]}")

# test accuracy of prediction
total_accuracy = 0
counter = 0
zeros_count_true = 0
ones_count_true = 0
zeros_count_pred = 0
ones_count_pred = 0

pred_zero_wrong = 0
pred_zero_correct = 0
pred_one_wrong = 0
pred_one_correct = 0

zeros_count_true_my = 0
ones_count_true_my = 0
zeros_count_pred_my = 0
ones_count_pred_my = 0

days_ahead = 4

model.eval()
with torch.no_grad():
    for t in range(375, 740):
        given_data = prcAll[:, :t]
        scaled_inputs = whole_data_prepare(given_data)
        predictions = predict(model, scaled_inputs)
        transformed_prediction = (prediction > 0.5).float() # shape = (50, 1)
        transformed_prediction = transformed_prediction.squeeze() # shape = (50)
        pred_prices = prcAll[:, t + days_ahead]
        curr_prices = prcAll[:, t]
        price_change = pred_prices - curr_prices
        price_change = torch.tensor(price_change)
        y_true = (price_change > 0).float()

        # see class ratio of true and predict
        num_ones_true = torch.sum(y_true).item()
        num_zeros_true = y_true.numel() - num_ones_true
        zeros_count_true += num_zeros_true
        ones_count_true += num_ones_true

        num_ones_pred = torch.sum(transformed_prediction).item()
        num_zeros_pred = transformed_prediction.numel() - num_ones_pred
        zeros_count_pred += num_zeros_pred
        ones_count_pred += num_ones_pred

        # calculate the accuracy of each class
        if len(y_true) == len(transformed_prediction):
            for row in range(len(y_true)):
                if y_true[row] == 1:
                    ones_count_true_my += 1
                    if transformed_prediction[row] == 1:
                        pred_one_correct +=1
                        ones_count_pred_my += 1
                    elif transformed_prediction[row] == 0:
                        pred_zero_wrong += 1
                        zeros_count_pred_my += 1
                    else:
                        print('hmm')
                elif y_true[row] == 0:
                    zeros_count_true_my += 1
                    if transformed_prediction[row] == 0:
                        pred_zero_correct += 1
                        zeros_count_pred_my += 1
                    elif transformed_prediction[row] == 1:
                        pred_one_wrong += 1
                        ones_count_pred_my += 1
                    else:
                        print('hmm')
                else:
                    print('how is it not 1 or 0')
        else:
            print('how is it not the same length')


        # calculate accuracy per prediction (50 predictions)
        correct = torch.eq(transformed_prediction, y_true)
        accuracy = torch.mean(correct.float())
        total_accuracy += accuracy
        counter += 1

print(f'\nmodel file: {model_file}')
print(f'days ahead predicting: {days_ahead}\n')
print('average accuracy: ', total_accuracy/counter)
print(f'pred ones ratio: {ones_count_pred}/{ones_count_pred + zeros_count_pred}, {round(ones_count_pred * 100/(ones_count_pred + zeros_count_pred), 2)}%')
print(f'true ones ratio: {ones_count_true}/{ones_count_true + zeros_count_true}, {round(ones_count_true * 100/(ones_count_true + zeros_count_true), 2)}%')
print(f'one pred accuracy: {round(pred_one_correct * 100 / (pred_one_correct + pred_one_wrong), 2)}%, ones total count: {pred_one_correct + pred_one_wrong}')
print(f'zero pred accuracy: {round(pred_zero_correct * 100 / (pred_zero_correct + pred_zero_wrong), 2)}%, zeros total count: {pred_zero_correct + pred_zero_wrong}')
# print(f'\nanother method check:')
# print(f'another average acc: {round((pred_zero_correct + pred_one_correct) / ((pred_zero_correct + pred_one_correct) + (pred_zero_wrong + pred_one_wrong)), 4)}')
# print(f'another pred ones ratio: {ones_count_pred_my}/{ones_count_pred_my + zeros_count_pred_my}, {round(ones_count_pred_my * 100/(ones_count_pred_my + zeros_count_pred_my), 2)}%')
# print(f'another true ones ratio: {ones_count_true_my}/{ones_count_true_my + zeros_count_true_my}, {round(ones_count_true_my * 100/(ones_count_true_my + zeros_count_true_my), 2)}%')

