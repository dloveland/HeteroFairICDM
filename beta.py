
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt


plt.figure()
markers = ['-', '--', '-.']
mean, var, skew, kurt = beta.stats(1, 1, moments='mvsk')
x = np.linspace(beta.ppf(0.01, 1, 1), beta.ppf(0.99, 1, 1), 100)
plt.plot(x, beta.pdf(x, 1, 1), 'g-', alpha=0.6, label='a=1, b=1')

for a in [1.5]:
    for i, b in enumerate([2, 5, 10]):
        mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        plt.plot(x, beta.pdf(x, a, b), f'r{markers[i]}', alpha=0.6, label=f'a={a}, b={b}')

for b in [1.5]:
    for i, a in enumerate([2,5, 10]):
        mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        plt.plot(x, beta.pdf(x, a, b), f'b{markers[i]}', alpha=0.6, label=f'a={a}, b={b}')
plt.legend()
plt.show()


plt.figure()

mean, var, skew, kurt = beta.stats(1, 1, moments='mvsk')
x = np.linspace(beta.ppf(0.01, 1, 1), beta.ppf(0.99, 1, 1), 100)
plt.plot(x, beta.pdf(x, 1, 1), 'g-', alpha=0.6, label='a=1, b=1')

for a in [20]:
    for i, b in enumerate([20, 50, 80]):
        mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        plt.plot(x, beta.pdf(x, a, b), f'r{markers[i]}', alpha=0.6, label=f'a={a}, b={b}')

for b in [100]:
    for i, a in enumerate([20, 50, 80]):
        mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        plt.plot(x, beta.pdf(x, a, b), f'b{markers[i]}', alpha=0.6, label=f'a={a}, b={b}')
plt.legend()
plt.show()

