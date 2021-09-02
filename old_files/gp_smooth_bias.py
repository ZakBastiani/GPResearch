from __future__ import division
import numpy as np
import torch
import matplotlib.pyplot as pl

# This is to test to find a smooth continuous function to represent the bias drifting over time


# This is the true unknown function we are trying to approximate

f = lambda x: (np.sin(0.9*x)).flatten()
# f = lambda x: (0.25*(x**2)).flatten()
bias = lambda x: (0.5*np.cos(x)).flatten()
# bias = lambda x: (0.1*np.sin(x)).flatten()

# Define the kernel for the true function
def true_function_kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 5
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

# Define the kernel for the smooth bias function
def bias_function_kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 8
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)


N = 10         # number of training points.
n = 50         # number of test points.
noise = 0.000005   # noise variance.
NTrue = 3      # number of true test points
alpha = 1   # Controlling the trust level of our bias predictions
# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + bias(X) + noise*np.random.randn(N)

# Also provide two true points, might have to choose the points for the variables
# Using random true points makes the testing inconsistent and hard to tell if there is an improvement
trueX = np.random.uniform(-3, 3, size=(NTrue,1))
# trueX = np.array([[-3], [0], [3]])
truey = f(trueX)

# First make a poor prediction of the actual points with only the given true data
K = true_function_kernel(trueX, trueX)
L = np.linalg.cholesky(K + noise*np.eye(NTrue))

Xtest = np.linspace(-5, 5, n).reshape(-1,1)

Lk = np.linalg.solve(L, true_function_kernel(trueX, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, truey))

K_ = true_function_kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

pl.figure(1)
pl.clf()
pl.plot(trueX, truey, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Poor Prediction using only the provided true points')
pl.axis([-5, 5, -3, 5])

# Subtract the predictions from (X,y) and solve for the potential bias
Lk = np.linalg.solve(L, bias_function_kernel(trueX, X))
mu = np.dot(Lk.T, np.linalg.solve(L, truey))

biasy = y - mu

K = true_function_kernel(X, X)
L = np.linalg.cholesky(K + noise*np.eye(N))

Xtest = np.linspace(-5, 5, n).reshape(-1,1)

Lk = np.linalg.solve(L, true_function_kernel(X, X))
smooth_bias = np.dot(Lk.T, np.linalg.solve(L, biasy))

Lk = np.linalg.solve(L, true_function_kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, biasy))

K_ = true_function_kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# Compare the bias with the actual bias function
pl.figure(2)
pl.clf()
pl.plot(X, biasy, 'r+', ms=20)
pl.plot(Xtest, bias(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Attempted guess at the bias function')
pl.axis([-5, 5, -3, 5])

# Removed the predicted bias from (X,y)
guessy = y - alpha * biasy

# then add the true points in the data
totalX = np.concatenate((X, trueX), axis=0)
totaly = np.concatenate((guessy, truey))

# find the final model
K = true_function_kernel(totalX, totalX)
L = np.linalg.cholesky(K + noise*np.eye(N + NTrue))

Xtest = np.linspace(-5, 5, n).reshape(-1,1)

Lk = np.linalg.solve(L, true_function_kernel(totalX, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, totaly))

K_ = true_function_kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# Compare the bias with the actual bias function
pl.figure(3)
pl.clf()
pl.plot(totalX, totaly, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.plot(Xtest, f(Xtest) + bias(Xtest), 'g-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.title('Final model of our attempted true function')
pl.axis([-5, 5, -3, 5])

X = np.concatenate((X, trueX), axis=0)
y = np.concatenate((y, truey))


K = true_function_kernel(X, X)
L = np.linalg.cholesky(K + noise*np.eye(N + NTrue))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, true_function_kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = true_function_kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(4)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Predictive model using all the data')
pl.axis([-5, 5, -3, 5])

# # draw samples from the prior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
# f_prior = np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(2)
# pl.clf()
# pl.plot(Xtest, f_prior)
# pl.title('Ten samples from the GP prior')
# pl.axis([-5, 5, -3, 3])
# pl.savefig('prior.png', bbox_inches='tight')
#
# # draw samples from the posterior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
# f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(3)
# pl.clf()
# pl.plot(Xtest, f_post)
# pl.title('Ten samples from the GP posterior')
# pl.axis([-5, 5, -3, 3])
# pl.savefig('post.png', bbox_inches='tight')

pl.show()
