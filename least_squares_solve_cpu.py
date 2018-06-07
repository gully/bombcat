#!/usr/bin/env python
import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
import argparse

parser = argparse.ArgumentParser(prog="least_sqaures_cpu.py", description="Demo of fitting a line with least squares.")
parser.add_argument("--Nsamples", type=int, default=5000)
args = parser.parse_args()

np.random.seed(123456)

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
true_theta = np.array([true_m, true_b])
N = args.Nsamples
x = np.linspace(-5, 5, N)
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties
#   and non-trivial correlated errors.
yerr = 0.1 + 0.4 * np.random.rand(N)
yerr_hom = 0.4*np.ones(N)
hom_cov = np.diag(yerr_hom ** 2)
iid_cov = np.diag(yerr ** 2)
true_cov = 0.5 * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + np.diag(yerr ** 2)

## This step draws the fake data from a MVN
## BUT it's TOO SLOW.  Do the hacky thing below instead.
#y = np.random.multivariate_normal(y, true_cov)
#np.save('y_fake.npy', y)
#y = np.load('y_fake.npy')

y = y + yerr*np.random.randn(N)
coeffs = np.load('correlation_coeffs.npy')
y = y + correlated_noise(x) # much faster!

#Linear algebra
A = np.vander(x, 2)
C = true_cov
guess_theta = np.array([true_m*1.1, true_b*0.9])
#model = np.dot(A, guess_theta)
model = np.dot(A, true_theta)

t0 = time.time()
factor, flag = cho_factor(C)
logdet = np.sum(2 * np.log((np.diag(factor))))
R = y - model
lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)
t1 = time.time()

net_time = t1-t0
print(" lnprob: {:.2f} \n time: {}".format(lnprob, net_time))