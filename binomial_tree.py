# implementing the binomial tree model for american option pricing
# approximate the BS-dynamics undeer the EMM

import numpy as np

def bi_tree_pricing_US(r, sigma, K, T=1, S0=1.0, N=100, p=0.5):
    """
    Binomial tree model for American put option pricing.
    """
    u = np.exp( (r - 0.5 * sigma**2 * T)/N + sigma * np.sqrt(0.5*T/N))
    d = np.exp( (r - 0.5 * sigma**2 * T)/N - sigma * np.sqrt(0.5 * T / N))
    if u-d < 0:
        N *= 2
        u = np.exp( (r - 0.5 * sigma**2 * T)/N + sigma * np.sqrt(0.5*T/N))
        d = np.exp( (r - 0.5 * sigma**2 * T)/N - sigma * np.sqrt(0.5*T/N))
        print(f"u-d < 0, N = {N}")
    
    # Initialisation
    option_vals = np.zeros(N + 1)
    for k in range(N + 1):
        S = S0 * (u ** k) * (d ** (N - k))
        option_vals[k] = max(K - S, 0)

    # Backward induction
    for n in range(N - 1, -1, -1):
        for k in range(n + 1):
            S = S0 * (u ** k) * (d ** (n - k))
            option_vals[k] = max(K - S, np.exp(-r * T / N) * (p * option_vals[k + 1] + (1 - p) * option_vals[k]))
    
    return option_vals[0]
    
# Plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r = 0.125  # Risk-free rate
    sigma = 0.5  # Volatility
    K = 1  # Strike price
    T = 1
    N = 100

    price_points = np.linspace(0.1, 2, N)

    payoff = K - price_points  # Payoff function

    # Compute binomial tree prices
    P_values = np.array([bi_tree_pricing_US(r, sigma, K, T, S, N, 0.5) for S in price_points])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(price_points, P_values, label="Binomial Tree Price")
    plt.plot(price_points, payoff, label="Payoff (K - S)", linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("American Put Option Pricing with Binomial Tree")
    plt.grid(True)
    plt.show()
    