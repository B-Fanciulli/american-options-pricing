# We want here to compare the performance of
# the Geske and Johnson against the trinomial method.

from GeskeJohnson import GeskeJohnsonMethod
from binomial_tree import bi_tree_pricing_US

import numpy as np
import matplotlib.pyplot as plt

T = 1
N = 100
time = np.linspace(0, T, N)
r = 0.125  # Risk-free rate
sigma = 0.5  # Volatility
K = 1  # Strike price
S0 = 0.5  # Initial price of the underlying asset

mod1 = GeskeJohnsonMethod(r, sigma, K)
GJ_values = np.array([mod1.richardson_extrapolation(S0, T-t) for t in time])

# Compute binomial tree prices
bi3_values = np.array([bi_tree_pricing_US(r, sigma, K, T-t, S0, N=100, p=0.5) for t in time])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, GJ_values, label="Geske-Johnson Method")
plt.plot(time, bi3_values, label="Binomial Tree Method")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Option Price")
plt.legend()
plt.title("Comparison of Option Pricing Methods")
plt.grid(True)
plt.show()

