import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools import whole_data_prepare_only_indicators



# printing graph for all 50 stocks over 500 days

# def loadPrices(fn):
#     global nt, nInst
#     df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
#     (nt, nInst) = df.shape
#     return (df.values).T


# pricesFile = "./prices750.txt"
# data = loadPrices(pricesFile)
# print("Loaded %d instruments for %d days" % (nInst, nt))

# plt.figure(figsize=(12, 6))
# for i in range(data.shape[0]):
#     plt.plot(data[i], alpha=0.5)

# plt.title('Stock Prices Over Time')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.show()

# data2 = data[:, :375]
# plt.figure(figsize=(12, 6))
# for i in range(data2.shape[0]):
#     plt.plot(data2[i], alpha=0.5)

# plt.title('Stock Prices Over Time')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.show()

# data3 = data[:, 500:]
# plt.figure(figsize=(12, 6))
# for i in range(data3.shape[0]):
#     plt.plot(data3[i], alpha=0.5)

# plt.title('Stock Prices Over Time')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.show()

#===================================================================

# finding how long take actual trend to appear

# df = pd.read_csv('stock_data_with_indicators.csv')
# average_change = []
# average_change_per_day = []
# for days_testing in range(1, 21):
#     total_change = 0
#     num_calculated = 0
#     for row_id in range(len(df) - days_testing - 1):
#         for column_id in range(50):
#             change = abs(df.iloc[row_id][column_id] - df.iloc[row_id + days_testing][column_id])
#             total_change += change
#             num_calculated += 1
#     average_change.append((days_testing, round(total_change/num_calculated, 4)))
#     average_change_per_day.append((days_testing, round((total_change/num_calculated)/days_testing, 4)))
            
# print(average_change)
# print(average_change_per_day)

#===================================================================

# finding number of ups and downs during different intervals, can load from all features or just prices, give same result, might need to change to df.iloc

# df = pd.read_csv('stock_data_with_indicators.csv')
# def loadPrices(fn):
#     global nt, nInst
#     df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
#     (nt, nInst) = df.shape
#     return (df.values).T

# pricesFile = "./prices.txt"
# df = loadPrices(pricesFile)
# df = df.transpose()
# print("Loaded %d instruments for %d days" % (nInst, nt))

# for days_testing in range(1, 21):
#     ups = 0
#     downs = 0
#     for row_id in range(len(df) - days_testing - 1):
#         for column_id in range(50):
#             change = df[row_id][column_id] - df[row_id + days_testing][column_id]
#             if change >= 0:
#                 downs += 1
#             else:
#                 ups += 1
           
#     print(f'day: {days_testing}, ups: {ups}, downs: {downs}, total: {ups + downs}')

#===================================================================

# finding best parameters to minimise for PL loss everyday

# with open('PL_hist_day680.pkl', 'rb') as file:
#     loaded_list = pickle.load(file)

# plt.figure(figsize=(12, 6))
# plt.plot(loaded_list)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()
# cumulative_sum = [sum(loaded_list[:i+1]) for i in range(len(loaded_list))]
# plt.plot(cumulative_sum)
# plt.show()

# # from_day375 = loaded_list[305:]
# from_day375 = loaded_list[:305]
# frequency_dict = {}
# negative_days = 0
# for dayPL in from_day375:
#     if dayPL <= 0:
#         negative_days += 1
#     else:
#         if negative_days != 0:
#             if negative_days not in frequency_dict.keys():
#                 frequency_dict[negative_days] = 1
#                 negative_days = 0
#             else:
#                 frequency_dict[negative_days] += 1
#                 negative_days = 0

# print(frequency_dict)
        

#===================================================================

# plotting each individual stock and its indicators 


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices750.txt"
data = loadPrices(pricesFile)
print(f'data.shape: {data.shape}')
print("Loaded %d instruments for %d days" % (nInst, nt))

processed_input = whole_data_prepare_only_indicators(data, 750)

# [SMA20, SMA50, SMA10, EMA20, EMA50, EMA10, RSI14, BB_middle, BB_upper, BB_lower, MACD, MACD_signal, stock_price_on_the_day]
#    1      2      3      4      5      6      7        8          9         10     11        12                  13

print(f'Processed data shape: {processed_input.shape}')
processed_input = np.transpose(processed_input, (0, 2, 1))
print(f'Processed data shape: {processed_input.shape}')

for counter, stock in enumerate(processed_input):
    plt.figure(figsize=(12, 6))
    for i in [0, 1, 2]:
        plt.plot(stock[i], alpha=0.5)
    # plot the price
    plt.plot(stock[12], color='red', linewidth=2)

    plt.title('SMA 10, 20, 50')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()
#==========================================
    plt.figure(figsize=(12, 6))
    for i in [3, 4, 5]:
        plt.plot(stock[i], alpha=0.5)
    # plot the price
    plt.plot(stock[12], color='red', linewidth=2)

    plt.title('EMA 10, 20, 50')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()
#==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Plot in the first subplot
    ax1.plot(stock[12], label='prices')
    ax1.set_title('Stock prices')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Prices')
    ax1.grid(True, alpha=0.3)

    # Plot in the second subplot
    for i in [6]:
        ax2.plot(stock[i], alpha=0.5)
        
    ax2.set_title('RSI 14')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()
#==========================================
    plt.figure(figsize=(12, 6))
    for i in [7,8,9]:
        plt.plot(stock[i], alpha=0.5)
    # plot the price
    plt.plot(stock[12], color='red', linewidth=2)

    plt.title('BB Upper, Lower, Middle')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()
#==========================================
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Plot in the first subplot
    ax1.plot(stock[12], label='prices')
    ax1.set_title('Stock prices')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Prices')
    ax1.grid(True, alpha=0.3)

    # Plot in the second subplot
    for i in [10,11]:
        ax2.plot(stock[i], alpha=0.5)
        
    ax2.set_title('MACD and MACD signal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()

    if counter == 0:
        break


#===================================================================

# finding how mcuh percent of the original price, the prices moves, in relation to their starting price. e.g penny stocks may have larger percentage return.

# finding stocks ranked by their volitility, maybe only trade stocks that move.

# finding profitability of each stock, then choose which stocks to get rid off, similar to above.