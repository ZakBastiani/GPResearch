from __future__ import division

import random
import numpy as np
import torch
import scipy.linalg
import matplotlib.pyplot as pl
from fqsmaster import fqs

# This is to test trying to find a GP while the majority of the provided data is know to have a bias
# In this first example we will know the normal distribution used to generate our bias, in future cases
# this will be unknown until proven.

# This is the true unknown function we are trying to approximate
f = lambda x: (np.sin(0.9*x)).flatten()
# f = lambda x: (0.25*(x**2)).flatten()

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 8
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)


# Calculate the bias matrix
def bias_matrix(data, sigma, variance_of_b, alpha, N):
    B = np.ndarray([N])
    alpha_matrix = [[]]
    variance = [[]]
    for i in range(0, N):
        alpha_matrix[0].append(alpha**2)
        variance[0].append(variance_of_b**2)
    holder = variance + (alpha_matrix @ sigma)

    for i in range(0,N):
        B[i] = ((variance_of_b**2)*data[i]/holder[0][i])

    return B


# Calculate alpha
def calc_alpha(data, sigma_inverse, B, N, alpha_mean, alpha_variance):
    # Faster method than np.roots
    roots = fqs.quartic_roots(
        [-(1/(alpha_variance**2)), (alpha_mean/(alpha_variance**2)), N, 0, ((data - B).T @ sigma_inverse @(data - B))])[0]
    # roots = np.roots(
    #   [-(1/(alpha_variance**2)), (alpha_mean/(alpha_variance**2)), N, 0, ((data - B).T @ sigma_inverse @ (data - B))])
    print("E: " + str(((data - B).T @ sigma_inverse @(data - B))))
    print("Roots: " + str(roots))
    # need to take the root closest to the mean and that doesnt have any imaginary parts
    for x in roots:
        if x.imag != 0:
            roots = roots[roots != x]

    return calc_closest(roots, alpha_mean)


# calculate the closest number from an array of complex numbers to a real number
# custom, some assumptions taken
def calc_closest(arr, num):
    closest = -10000
    distance = 10000
    for x in arr:
        if np.abs(x.real - num) < distance:
            closest = x.real
            distance = np.abs(x-num)
    return closest


N = 10         # number of training points.
n = 50         # number of test points.
noise = 0.0005     # noise variance.
NTrue = 2      # number of true test points
alpha = 1
variance_of_b = 0.25

# Sample some input points and noisy versions of the function evaluated at
# these points.
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + np.random.normal(0, 1, size=(N)) + noise*np.random.randn(N)

# Also provide two true points
trueX = np.random.uniform(-3, 3, size=(NTrue,1))
truey = f(trueX)

# Should first match Wei's method to predict
# Build a model using hyper parameters alpha and b.
# Then solve for a minimized solution of the system,
# where alpha and b are also given a predefined probability distribution

K = kernel(X, X)
alpha_current = alpha
alpha_previous = alpha - 3
while abs(alpha_current - alpha_previous) > 0.001:
    alpha_previous = alpha_current
    B = bias_matrix(y, K, variance_of_b, alpha_previous, N)
    alpha_current = calc_alpha(y, np.linalg.inv(K), B, N, 1, 1) # should be using the inverse of K instead of K
    print("alpha: " + str(alpha_current))
    print("B: " + str(B))
alpha = alpha_current
print("alpha: " + str(alpha))

guessY = y - B/alpha**2
L_ba = np.linalg.cholesky(K + noise*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk_ba = np.linalg.solve(L_ba, kernel(X, Xtest))
mu_ba = np.dot(Lk_ba.T, np.linalg.solve(L_ba, guessY))

# compute the mean at our test points without bias and without alpha.
L = np.linalg.cholesky(K + noise*np.eye(N))
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
pl.plot(Xtest, mu_ba, 'g--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Guessing Random Bias without True points')
pl.axis([-5, 5, -3, 5])

# then add the true points in the data
totalX = np.concatenate((X, trueX), axis=0)
totaly = np.concatenate((guessY, truey))

# find the final model
K = kernel(totalX, totalX)
L = np.linalg.cholesky(K + noise*np.eye(N+NTrue))

Xtest = np.linspace(-5, 5, n).reshape(-1,1)

Lk = np.linalg.solve(L, kernel(totalX, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, totaly))

K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

pl.figure(2)
pl.clf()
pl.plot(X, guessY, 'g+', ms=20)
pl.plot(trueX, truey, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Final model of our attempted true function')
pl.axis([-5, 5, -3, 5])
pl.show()

