import pandas as pd
import numpy as np

def load_data(file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n')
    
    # Convert the data to a 2D numpy array
    data_array = np.array([list(map(float, row.split())) for row in data])
    
    print(f'data shape: {data_array.shape}')
    
    return data_array

def calculate_indicators(data):
    # Number of days and stocks
    num_days, num_stocks = data.shape
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data, columns=[f'Stock_{i+1}' for i in range(num_stocks)])
    
    # Calculate indicators for each stock
    print(len(df.columns))
    print(df.columns)
    for stock in df.columns:
        
        # Simple Moving Average (SMA) - 10/20/50 days
        df[f'{stock}_SMA20'] = df[stock].rolling(window=20).mean()
        df[f'{stock}_SMA50'] = df[stock].rolling(window=50).mean()
        df[f'{stock}_SMA10'] = df[stock].rolling(window=10).mean()
        
        # Exponential Moving Average (EMA) - 10/20/50 days
        df[f'{stock}_EMA20'] = df[stock].ewm(span=20, adjust=False).mean()
        df[f'{stock}_EMA50'] = df[stock].ewm(span=50, adjust=False).mean()
        df[f'{stock}_EMA10'] = df[stock].ewm(span=10, adjust=False).mean()
        
        # Relative Strength Index (RSI) - 14 days
        delta = df[stock].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'{stock}_RSI14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - 20 days, 2 standard deviations
        df[f'{stock}_BB_middle'] = df[stock].rolling(window=20).mean()
        df[f'{stock}_BB_upper'] = df[f'{stock}_BB_middle'] + 2 * df[stock].rolling(window=20).std()
        df[f'{stock}_BB_lower'] = df[f'{stock}_BB_middle'] - 2 * df[stock].rolling(window=20).std()
        
        # Moving Average Convergence Divergence (MACD)
        exp1 = df[stock].ewm(span=12, adjust=False).mean()
        exp2 = df[stock].ewm(span=26, adjust=False).mean()
        df[f'{stock}_MACD'] = exp1 - exp2
        df[f'{stock}_MACD_signal'] = df[f'{stock}_MACD'].ewm(span=9, adjust=False).mean()
        break
    df = df.fillna(0)
    df = df.round(2)
    return df

def main():
    # Load the data
    data = load_data('prices.txt')
    
    # Calculate indicators
    df_with_indicators = calculate_indicators(data)
    
    # Save the results to a CSV file
    # df_with_indicators.to_csv('stock_data_with_indicators.csv', index=False)
    # print("Data with indicators has been saved to 'stock_data_with_indicators.csv'")

if __name__ == "__main__":
    main()