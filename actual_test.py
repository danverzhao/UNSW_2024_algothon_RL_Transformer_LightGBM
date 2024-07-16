import pandas as pd
import numpy as np

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

# Example usage
arr = np.array([
    [1, 2, 3, 4, 98],
    [6, 7, 8, 9, 99],
    [11, 12, 13, 14, 100]
])

print("Original DataFrame:")
print(arr)

# Normalize columns A and B together
normalized_df = global_minmax_normalize(arr, [1,2,4])

print("\nNormalized DataFrame (A and B normalized together, C unchanged):")
print(normalized_df)


print(f'flat: {arr.flatten()} {arr.flatten().shape}')