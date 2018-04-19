#!/usr/bin/env python
import numpy as np
import time
import argparse
from scipy.linalg import cho_factor, cho_solve

parser = argparse.ArgumentParser(prog="least_sqaures_cpu.py", description="Demo of fitting a line with least squares.")
parser.add_argument("--Nsamples", type=int, default=5000)
args = parser.parse_args()

np.random.seed(123456)

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
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
y_noised = np.random.multivariate_normal(y, true_cov)
#np.save('y_fake.npy', y)
#y = np.load('y_fake.npy')

#Linear algebra
A = np.vander(x, 2)
AT= A.T
C = iid_cov

t0 = time.time()
factor = cho_factor(C, overwrite_a=True)
S_inv = np.dot(AT, cho_solve(factor, A))
S = np.linalg.inv(S_inv)
ls_m, ls_b = np.dot(S, np.dot(AT, cho_solve(factor, y_noised)))
t1 = time.time()

net_time = t1-t0
print(" m: {:.2f} \n b: {:.2f} \n time: {}".format(ls_m, ls_b, net_time))
