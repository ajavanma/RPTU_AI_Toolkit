"""Interpret the results:

    Skewness: a value close to 0 indicates a symmetric distribution. Positive values indicate right-skew, while negative values indicate left-skew.
    Kurtosis: a value close to 0 indicates a distribution with similar tails to a normal distribution. Positive values indicate heavy tails, while negative values indicate light tails.
    Histogram and Q-Q plot: Visually inspect these plots for deviations from normality.
    Statistical tests: A low p-value (typically below 0.05) indicates that the data is not normally distributed.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.read_csv('raw_file.csv')

column = data['x']

skewness = column.skew()
kurtosis = column.kurt()

plt.figure(figsize=(10,5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(column, bins=30, color='blue', edgecolor='k', alpha=0.7)
plt.title('Histogram')

# Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(column, dist='norm', plot=plt)
plt.title('Q-Q Plot')

plt.show()


# Shapiro-Wilk test
shapiro_test = stats.shapiro(column)
print('Shapiro-Wilk test statistic:', shapiro_test[0])
print('Shapiro-Wilk p-value:', shapiro_test[1])

# Kolmogorov-Smirnov test
ks_test = stats.kstest(column, 'norm')
print('Kolmogorov-Smirnov test statistic:', ks_test.statistic)
print('Kolmogorov-Smirnov p-value:', ks_test.pvalue)
