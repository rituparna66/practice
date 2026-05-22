import numpy as np
import matplotlib.pyplot as plt

# Simulate rolling two dice 10,000 times
n = 10000

dice1 = np.random.randint(1, 7, size=n)
dice2 = np.random.randint(1, 7, size=n)

# Sum of the two dice
sums = dice1 + dice2

# Count frequencies using NumPy only
freq = np.bincount(sums)[2:]   # sums range from 2 to 12

# Possible sums
values = np.arange(2, 13)

# Plot frequency distribution
plt.figure(figsize=(8,5))
plt.bar(values, freq)

plt.xlabel("Dice Sum")
plt.ylabel("Frequency")
plt.title("Distribution of Two Dice Sums (10,000 Rolls)")

plt.show()
