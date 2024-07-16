
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import StandardScaler
from tools import whole_data_prepare, calculate_current_position, get_max_position_for_each, get_all_PL
import pickle
from RL import QNetwork, DQNAgent
from tools import whole_data_prepare_only_indicators, global_minmax_normalize
import os

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst) # I need to maintain positions for all stocks at all times, so after calculate change of holdings
                             # I need to apply to this
days = 0
days_for_each = [0, -1, -2, -3, -4, -5, -6, -7, -8]
preds_for_each = [0, 0,  0,  0,  0,  0,  0,  0,  0]
all_pl = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

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



#======================================================
# For transformers
# Hyperparameters
# d_model = 128
# nhead = 4
# num_layers = 2
# dim_feedforward = 128
# input_dim = 63


# file_path = '3days_best_model_0.5724.pth'
# loaded_transformer = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward)
# loaded_transformer.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
# loaded_transformer.eval()


#======================================================
# For RL DQN

def load_dqn_model(model_path, state_size, action_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = QNetwork(state_size, action_size).to(device)
    
    # Load the saved state
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        
        print(f"Model loaded from episode {episode}")
        
        # Initialize the DQNAgent with the loaded model
        agent = DQNAgent(state_size, action_size)
        agent.model = model
        agent.epsilon = 0 # 
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent, episode
    else:
        print(f"No saved model found at {model_path}")
        return None, None


ep_num = 750
model_path = f"dqn_model_episode_{ep_num}.pth"
loaded_agent, loaded_episode = load_dqn_model(model_path, state_size=650, action_size=3)
#======================================================    

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
        
        today_price = df_prices.iloc[-seq_length:][stock]
        
        # print(f'SMA: {SMA10} {SMA20} {SMA50}')
        # print(f'EMA: {EMA10} {EMA20} {EMA50}')
        # print(f'RSI: {RSI14}')
        # print(f'BB: {BB_lower} {BB_middle}, {BB_upper}')
        # print(f'MACD: {MACD} {MACD_signal}')
        
        final_list.append(np.array([SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal, today_price]))
    
    return final_list



def solution1_consecutive_days(prices, consecutive_days):
    stock_num = 0
    # this shows how many days of stock needs to line up.
    result_pos = np.zeros(nInst)
    prices = prices[:, -consecutive_days:]
    
    for stock in prices:
        stock = stock.tolist()
        if stock == sorted(stock):
            result_pos[stock_num] = 100
        
        elif stock == sorted(stock, reverse=True):
            result_pos[stock_num] = -100
        
        else:
            result_pos[stock_num] = 0
            
        stock_num += 1
    
    return result_pos
    
def solution2_lightgbm_1stock_all_indicators(df_prices, price_diff):
    
    result_pos = np.zeros(nInst)
    print('hi')
    model_filename = 'lightgbm_model_1stock_and_all_indicators.joblib'
    lightgbm = joblib.load(model_filename)
    print('2')
    inputs = get_features_overtime(df_prices, 1)
    
    for stock in range(50):
        input = inputs[stock]
        input = input.round(2)
        pred = lightgbm.predict(input)
        today_price = input[0][-1]
        
        if (pred - today_price) > price_diff:
            result_pos[stock] = 1000
        elif (today_price - pred) > price_diff:
            result_pos[stock] = -1000
        else:
            result_pos[stock] = 0

    return result_pos

def solution3_solution_1and2_ensemble(df_price, consecutive_days, price_diff):
    result_pos = np.zeros(nInst)
    
    result1 = solution1_consecutive_days(df_price, consecutive_days=consecutive_days)
    result2 = solution2_lightgbm_1stock_all_indicators(df_price, price_diff=price_diff)
    
    
    for i in range(nInst):
        if result1[i] == 0 or result2[i] == 0:
            result_pos[i] = 0
        elif result1[i] > 0 and result2[i] > 0:
            result_pos[i] = 1000
        elif result1[i] < 0 and result2[i] < 0:
            result_pos[i] = -1000
        else:
            result_pos[i] = 0
    
    return result_pos
'''
def solution4_transformer(df_price, curPos):

    # e.g. (7, 4) means: if days % 7 in [x for x in range(1, 4)]:
    # avg (10, 27.25) means: in eval.py there's 10 starting dates consecutively from day 375, and average score of 27.25

    # days 375 - 501
    #                                                (3, 2) -4.97, avg (10, 2.07)
    #                (4, 2) 3.59                     (4, 3) 15.23, avg (10, 15.4)
    #                (5, 3) 8.33                     (5, 4) 20.69, avg (10, 27.25)
    # (6, 3) -5.93   (6, 4) 1.88,  avg (10, 21.56)   (6, 5) 3.14,  avg (10, 18.35)
    # (7, 4) 4.31    (7, 5) 12.52, avg (10, 25.01)   (7, 6) 26.22, avg (10, 33.41)   (7, 7) 22.31, avg (10, 34.33)  
    #                (8, 6) 12.52, avg (10, 24.54)   (8, 7) 32.83, avg (10, 29.37) 
    # (9, 6) 6.15    (9, 7) 18.44                    (9, 8) -0.45, avg (10, 17.94)

    # days 70 - 501
    #                               (4, 3) 73.60
    #                               (5, 4) 81.67
    #                (6, 4) 75.51   (6, 5) 77.71   
    # (7, 4) xxxx    (7, 5) 70.05   (7, 6) 76.79,  avg (10, 76.17)
    #                (8, 6) 78.66   (8, 7) 80.65   
    # (9, 6) xxxx    (9, 7) xxxxx   (9, 8) 72.12

    # days 70 - 375 (for fun)
    # (7, 7) avg (10, 91.48)
    if days % 7 in [x for x in range(1, 7)]:
        return curPos
    inputs = whole_data_prepare(df_price, seq_length=50)
    
    loaded_transformer.eval()
    with torch.no_grad():
        
        result_pos = np.zeros(nInst)
        predictions = loaded_transformer(inputs)
        transformed_prediction = (predictions > 0.5).float() # shape = (50, 1)
        transformed_prediction = transformed_prediction.squeeze() # shape = (50)

        for row in range(len(transformed_prediction)):
            if transformed_prediction[row] == 1:
                pass
            elif transformed_prediction[row] == 0:
                result_pos[row] = -1000

        return result_pos
                

def solution5_transformer_avg(df_price, all_pl):
    
    # plan: since the model either give max or 0 we can have voting
    # output if a stock is being shorted (-1) or not (0) for each active predictor
    # return ratio of votes of the max position of each stock 

    # consideration, do you use the maxPos or curPos to scale with the votes
    # try maxPos then curPos


    total_votes_stock = np.zeros(nInst)
    num_valid_vote = 0
    days_ahead_to_pred = 5
    for model_number in range(days_ahead_to_pred):
        day = days_for_each[model_number]
        if day >= 0:
            num_valid_vote += 1
            vote = solution5_model_helper(df_price, model_number, days_ahead_to_pred)
            total_votes_stock += vote
    
    ratio_votes_stock = total_votes_stock/num_valid_vote
    max_position_for_each = get_max_position_for_each(df_price)
   
    # for stock in range(len(ratio_votes_stock)):
    #     if ratio_votes_stock[stock] < -0.2:
    #         ratio_votes_stock[stock] = -1
    #     else:
    #         ratio_votes_stock[stock] = 0
    result_pos = max_position_for_each * ratio_votes_stock
    
    return result_pos
        


def solution5_model_helper(df_price, model_number, days_ahead_to_pred):
    global preds_for_each
    days = days_for_each[model_number]
    previous_pred = preds_for_each[model_number]

    if days % days_ahead_to_pred in [x for x in range(1, days_ahead_to_pred)]:
        return previous_pred
    
    inputs = whole_data_prepare(df_price, seq_length=50)
    
    loaded_transformer.eval()
    with torch.no_grad():
        
        result_pos = np.zeros(nInst)
        predictions = loaded_transformer(inputs)
        transformed_prediction = (predictions > 0.001).float() # shape = (50, 1)
        transformed_prediction = transformed_prediction.squeeze() # shape = (50)

        for row in range(len(transformed_prediction)):
            if transformed_prediction[row] == 1:
                pass
            elif transformed_prediction[row] == 0:
                result_pos[row] = -1

        preds_for_each[model_number] = result_pos
        return result_pos
'''
def pl_loss_minimiser(cur_pos, all_pl):
    all_pl.reverse()
    num_days_negative = 0
    neg_pl_list = []
    
    for day_PL in all_pl:
        if day_PL > 0:
            break
        else:
            neg_pl_list.append(day_PL)
            num_days_negative += 1
    
    # if sum(neg_pl_list) < -500:
    #     return cur_pos / -
    if num_days_negative > 4:
        return -cur_pos #/ (num_days_negative * 1) 
    else:
        return cur_pos


def solution6_RL(prices):
    global currentPos
    result_pos = np.zeros(nInst)
    for counter, stock in enumerate(prices):
        inputs = whole_data_prepare_only_indicators(stock, seq_length=50)
        inputs = np.squeeze(inputs, axis=0)
        inputs = global_minmax_normalize(inputs, columns_to_normalize=[0,1,2,3,4,5,7,8,9,12])
        inputs = inputs.flatten()
        with torch.no_grad():
            action_values = loaded_agent.model(torch.Tensor(inputs).to(loaded_agent.device))
        predicted_action = torch.argmax(action_values).item()
        if predicted_action == 0:  # Sell
            result_pos[counter] = -100000

        elif predicted_action == 1: # Hold
            result_pos[counter] = currentPos[counter]
        
        elif predicted_action == 2:  # Buy
            result_pos[counter] = 100000

    return result_pos


def getMyPosition(prcSoFar):
    global currentPos
    global days
    global days_for_each
    global all_pl
    
    
    # result_pos = solution1_consecutive_days(prcSoFar, consecutive_days=7)
    # result_pos = solution2_lightgbm_1stock_all_indicators(prcSoFar, price_diff=1)
    # result_pos = solution3_solution_1and2_ensemble(prcSoFar, consecutive_days=5, price_diff=1.1) 
    # result_pos = solution4_transformer(prcSoFar, currentPos)
    # result_pos = solution5_transformer_avg(prcSoFar, all_pl)
    
    
    result_pos = solution6_RL(prcSoFar)
    currentPos = calculate_current_position(result_pos, prcSoFar)
    all_pl = get_all_PL(currentPos, prcSoFar)
    
    # if len(all_pl) > 0:
    #     result_pos = pl_loss_minimiser(currentPos, all_pl)


    days += 1
    days_for_each = [element + 1 for element in days_for_each]

    return currentPos