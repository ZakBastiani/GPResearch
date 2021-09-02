from __future__ import division

import random

import numpy as np
import torch
import matplotlib.pyplot as pl

# This is to test trying to find a GP while the majority of the provided data is know to have a bias
# In this first example we will know the normal distribution used to generate our bias, in future cases
# this will be unknown until proven.

# This is the true unknown function we are trying to approximate
f = lambda x: (np.sin(0.9*x)).flatten()
#f = lambda x: (0.25*(x**2)).flatten()

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 8
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)


# Calculate the bias matrix
def bias_matrix(data, sigma, N):
    B = []
    one_by_n = [[]]
    for i in range(0, N):
        one_by_n[0].append(1)
    holder = one_by_n + (one_by_n @ sigma)

    for i in range(0,N):
        B.append(data[i]/holder[0][i])

    return B


N = 10         # number of training points.
n = 50         # number of test points.
s = 0.0005     # noise variance.
NTrue = 2      # number of true test points

# Sample some input points and noisy versions of the function evaluated at
# these points.
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + np.random.normal(0, 1, size=(N)) + s*np.random.randn(N)

# Also provide two true points
trueX = np.random.uniform(-3, 3, size=(NTrue,1))
truey = f(trueX)

# Should first match Wei's method to predict
# Build a model using hyper parameters alpha and b.
# Then solve for a minimized solution of the system,
# where alpha and b are also given a predefined probability distribution

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))
B = bias_matrix(y, K, N)
guessY = y - B  # you messed up somewhere this is supposed to be a minus
# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk_b = np.linalg.solve(L, kernel(X, Xtest))
mu_b = np.dot(Lk_b.T, np.linalg.solve(L, guessY))

# compute the mean at our test points without bias.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(X, guessY, 'g+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.plot(Xtest, mu_b, 'g--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Guessing Random Bias without True points')
pl.axis([-5, 5, -3, 5])

pl.show()
