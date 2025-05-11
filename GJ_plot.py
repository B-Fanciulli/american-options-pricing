import matplotlib.pyplot as plt
import numpy as np
from GeskeJohnson import GeskeJohnsonMethod

r = 0.125  # Risk-free rate
sigma = 0.5  # Volatility
K = 1  # Strike price
T = 1
N = 100
mod1 = GeskeJohnsonMethod(r, sigma, K)
price_points = np.linspace(0.1, 2, N)

payoff = K - price_points  # Payoff function

# Compute P1 values
P1_values = np.array([mod1.P1(S, T) for S in price_points])

# Compute P2 values
P2_values = np.array([mod1.P2(S, T) for S in price_points])

# Compute P3 values
P3_values = np.array([mod1.P3(S, T) for S in price_points])

# Compute Richardson extrapolation
richardson_extrapolation = np.array([mod1.richardson_extrapolation(S, T) for S in price_points])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(price_points, P1_values, label="P1 (European Put)")
plt.plot(price_points, P2_values, label="P2 (Twice Exercisable Put)")
plt.plot(price_points, P3_values, label="P3 (Thrice Exercisable Put)")
plt.plot(price_points, richardson_extrapolation, label="Richardson Extrapolation", linestyle=':')
plt.plot(price_points, payoff, label="Payoff (K - S)", linestyle='--')
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.xlabel("Stock Price (S)")
plt.ylabel("Value")
plt.legend()
plt.title("Option Pricing with Geske-Johnson Method")
plt.grid(True)
plt.show()
