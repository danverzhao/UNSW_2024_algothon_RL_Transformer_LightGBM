import numpy as np

# Create sample data
# Stock prices: 100, 101, 102, ..., 109
prices = np.arange(100, 110)

# Indicator values: 0.50, 0.51, 0.52, ..., 0.59
indicators = np.arange(50, 60) / 100

# Combine prices and indicators
data = np.column_stack((prices, indicators))

# Prepare input sequences
sequence_length = 4  # Number of days to look back
X = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])

X = np.array(X)

print("Input shape (X):", X.shape)
print("\nX (input sequences):")
for i, sequence in enumerate(X):
    print(f"\nSequence {i + 1}:")
    print("  Day | Price | Indicator")
    print("  ------------------------")
    for day, (price, indicator) in enumerate(sequence, 1):
        print(f"  {day:3d} | {price:5.2f} | {indicator:.2f}")

print("\nExplanation of X shape:")
print(f"- Number of sequences: {X.shape[0]}")
print(f"- Sequence length: {X.shape[1]} days")
print(f"- Features per day: {X.shape[2]} (price and indicator)")

print(f'\n{X}')

print(f'\n{data}')
