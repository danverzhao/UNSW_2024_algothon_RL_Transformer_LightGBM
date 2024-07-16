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
        
        custom_input = [SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal]

        column_series_list = [all_price[column] for column in all_price.columns]

       
        for stock_prices in column_series_list:
            custom_input.append(stock_prices)

        series_of_stock_id = pd.Series([stock+1] * seq_length)
        custom_input.append(series_of_stock_id)

        final_list.append(np.array(custom_input))

        
    return np.array(final_list)

def whole_data_prepare(df_price, seq_length):

    inputs = get_features_overtime(df_price, seq_length)

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



nInst = 50
nt = 500
commRate = 0.0010
dlrPosLimit = 10000
curPos = np.zeros(nInst)

def calculate_current_position(generated_position_by_me, current_price):
    global curPos
    cash = 0
    totDVolume = 0
    
    prcHistSoFar = current_price # (50, 250) <class 'numpy.ndarray'>
    newPosOrig = generated_position_by_me
    curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
    posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # clip order limit
    newPos = np.clip(newPosOrig, -posLimits, posLimits)
    deltaPos = newPos - curPos
    dvolumes = curPrices * np.abs(deltaPos)
    dvolume = np.sum(dvolumes)
    totDVolume += dvolume # total money traded
    comm = dvolume * commRate # commision for trading on the day
    cash -= curPrices.dot(deltaPos) + comm
    curPos = np.array(newPos)

    return curPos

def get_max_position_for_each(current_price):
    cash = 0
    totDVolume = 0
    
    prcHistSoFar = current_price # (50, 250) <class 'numpy.ndarray'>
    newPosOrig = np.full(50, 100000)
    curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
    posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # clip order limit
    newPos = np.clip(newPosOrig, -posLimits, posLimits)
    deltaPos = newPos
    dvolumes = curPrices * np.abs(deltaPos)
    dvolume = np.sum(dvolumes)
    totDVolume += dvolume # total money traded
    comm = dvolume * commRate # commision for trading on the day
    cash -= curPrices.dot(deltaPos) + comm
    curPos = np.array(newPos)

    return curPos

todayPLL1 = []
cash1 = 0
curPos1 = np.zeros(50)
totDVolume1 = 0
value1 = 0
commRate1 = 0.0010
dlrPosLimit1 = 10000


def get_all_PL(my_pos, current_price):
    global todayPLL1
    global cash1 
    global curPos1
    global totDVolume1
    global value1
    global commRate1
    global dlrPosLimit1

    
    prcHistSoFar = current_price # (50, 250) <class 'numpy.ndarray'>
    newPosOrig = my_pos
    curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
    posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # clip order limit
    newPos = np.clip(newPosOrig, -posLimits, posLimits)
    deltaPos = newPos - curPos1
    dvolumes = curPrices * np.abs(deltaPos)
    dvolume = np.sum(dvolumes)
    totDVolume1 += dvolume # total money traded
    comm = dvolume * commRate1 # commision for trading on the day
    cash1 -= curPrices.dot(deltaPos) + comm
    curPos1 = np.array(newPos)
    posValue = curPos1.dot(curPrices)
    todayPL = cash1 + posValue - value1
    todayPLL1.append(todayPL)
    value1 = cash1 + posValue
    ret = 0.0
    if (totDVolume1 > 0):
        ret = value1 / totDVolume1
    
    return todayPLL1
    
#===========================================================================================


# returns a torch.Size([number of stocks, seq_length, 13 features])
# [SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal, stock_price_on_the_day]
#    1      2      3      4      5      6      7        8          9         10     11        12                  13
#    0      1      2      3      4      5      6        7          8         9      10        11                  12
def get_features_overtime_1stock_only_indicators(df_prices, seq_length):
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
        
        all_price = df_prices.iloc[-seq_length:][stock]
        
        custom_input = [SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal]

        #column_series_list = [all_price[column] for column in all_price.columns]

       
        #for stock_prices in column_series_list:
        custom_input.append(all_price)


        final_list.append(np.array(custom_input))

        
    return np.array(final_list)




def whole_data_prepare_only_indicators(df_price, seq_length):
    
    inputs = get_features_overtime_1stock_only_indicators(df_price, seq_length)

    inputs = np.transpose(inputs, (0, 2, 1))


    # Prepare your input data
    def normalise_input_data(raw_data):
        # Assuming raw_data is a numpy array of shape (sequence_length, input_dim)
        # making a new scaler
        scaler = StandardScaler()
        raw_data_reshaped = raw_data.reshape(-1, raw_data.shape[-1])
        raw_data_scaled = scaler.fit_transform(raw_data_reshaped).reshape(raw_data.shape)

        return torch.FloatTensor(raw_data_scaled)  # Add batch dimension

    return inputs

    #scaled_inputs = normalise_input_data(inputs)

    #return scaled_inputs


def global_minmax_normalize(arr, columns_to_normalize):
    df = pd.DataFrame(arr)
    # Extract the selected columns
    selected_data = df[columns_to_normalize]
    
    # Find the global min and max across all selected columns
    global_min = selected_data.values.min()
    global_max = selected_data.values.max()
    
    # Apply normalization
    df[columns_to_normalize] = (selected_data - global_min) / (global_max - global_min)
    
    return df.to_numpy()