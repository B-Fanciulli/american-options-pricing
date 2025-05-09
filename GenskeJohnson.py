

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import newton

class GeskeJohnsonMethod:
    def __init__(self, r, sigma, K):
        """
        Initialize the Geske-Johnson method parameters.

        Parameters:
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility of the underlying asset.
        K : float
            Strike price of the option.
        S0 : float
            Initial price of the underlying asset.
        """
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.K = K  # Strike price
        self.rho12 = np.sqrt(1 / 2)  # Correlation coefficient between first and second steps
        self.rho13 = np.sqrt(1 / 3)  # Correlation coefficient between first and third steps
        self.rho23 = np.sqrt(2 / 3)  # Correlation coefficient between second and third steps

    def beta(self, dt):
        """Discounting factor."""
        return np.exp(-self.r * dt)

    def d1(self, S, q, dt):
        """
        d1 term in Black-Scholes formula.

        Parameters:
        S : float
            Current stock price.
        q : float
            Critical price.
        dt : float
            Time elapsed.
        """
        return (np.log(S / q) + (self.r + 0.5 * self.sigma ** 2) * dt) / (self.sigma * np.sqrt(dt))

    def d2(self, S, q, dt):
        """d2 term in Black-Scholes formula."""
        return self.d1(S, q, dt) - self.sigma * np.sqrt(dt)

    def P1(self, S, dt):
        """
        Price of a European put option.

        Parameters:
        S : float
            Current stock price.
        dt : float
            Time to maturity.
        """
        return self.K * self.beta(dt) * norm.cdf(-self.d2(S, self.K, dt)) - S * norm.cdf(-self.d1(S, self.K, dt))

    def critical_price(self, P, dt):
        """
        Find the critical price for a given pricing function.

        Parameters:
        P : callable
            Pricing function (e.g., self.P1 or self.P2).
        dt : float
            Time to maturity.
        """
        func = lambda S: self.K - S - P(S, dt)
        return newton(func, x0=self.K)

    def P2(self, S, dt):
        """
        Price of the twice-exercisable put option.

        Parameters:
        S : float
            Stock price.
        dt : float
            Time to maturity.
        """
        S_crit = self.critical_price(self.P1, dt / 2)
        # Term 1: Exercise at T/2
        term1 = self.K * self.beta(dt / 2) * norm.cdf(-self.d2(S, S_crit,  dt / 2)) \
                - S * norm.cdf(-self.d1(S, S_crit, dt / 2))

        # Term 2: Exercise at T
        d2_crit = self.d2(S, S_crit, dt / 2)
        d1_crit = self.d1(S, S_crit, dt / 2)

        d2_final = self.d2(S, self.K, dt)
        d1_final = self.d1(S, self.K, dt)

        # Compute multivariate normal CDF
        mean = [0, 0]  # Mean of the bivariate normal
        cov = [[1, -self.rho12], [-self.rho12, 1]]  # Covariance matrix
        term2 = self.K * self.beta(dt) * multivariate_normal.cdf([d2_crit, -d2_final], mean, cov) \
                - S * multivariate_normal.cdf([d1_crit, -d1_final], mean, cov)

        return term1 + term2

    def P3(self, S, dt):
        """
        Price of the thrice-exercisable put option.

        Parameters:
        S : float
            Stock price.
        dt : float
            Time to maturity.
        """
        S_crit_1 = self.critical_price(self.P2, 2 * dt / 3)
        S_crit_2 = self.critical_price(self.P1, dt / 3)

        # Term 1: Exercise at T/3
        term1 = self.K * self.beta(dt / 3) * norm.cdf(-self.d2(S, S_crit_1, dt / 3)) \
                - S * norm.cdf(-self.d1(S, S_crit_1, dt / 3))

        # Term 2: Exercise at 2T/3
        mean_2 = [0, 0]
        cov_2 = [[1, -self.rho12], [-self.rho12, 1]]
        term2 = self.K * self.beta(2 * dt / 3) * multivariate_normal.cdf([self.d2(S, S_crit_1, dt / 3), -self.d2(S, S_crit_2, 2 * dt / 3)], mean_2, cov_2) \
                - S * multivariate_normal.cdf([self.d1(S, S_crit_1, dt / 3), -self.d1(S, S_crit_2, 2 * dt / 3)], mean_2, cov_2)

        # Term 3: Exercise at T
        mean_3 = [0, 0, 0]
        cov_3 = [[1, self.rho12, -self.rho13], [self.rho12, 1, -self.rho23], [-self.rho13, -self.rho23, 1]]
        term3 = self.K * self.beta(dt) * multivariate_normal.cdf([self.d2(S, S_crit_1, dt / 3), self.d2(S, S_crit_2, 2 * dt / 3), -self.d2(S, self.K, dt)], mean_3, cov_3) \
                - S * multivariate_normal.cdf([self.d1(S, S_crit_1, dt / 3), self.d1(S, S_crit_2, 2 * dt / 3), -self.d1(S, self.K, dt)], mean_3, cov_3)

        return term1 + term2 + term3

    def richardson_extrapolation(self, S, dt):
        """
        Richardson extrapolation to approximate the option price.

        Parameters:
        S : float
            Stock price.
        dt : float
            Time to maturity.
        """
        P1 = self.P1(S, dt)
        P2 = self.P2(S, dt)
        P3 = self.P3(S, dt)

        return P3 + (7 / 2) * (P3 - P2) - (1 / 2) * (P2 - P1)


