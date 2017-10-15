#!/usr/bin/env python
import numpy as np
import time
import argparse
import torch

parser = argparse.ArgumentParser(prog="least_sqaures_gpu.py", description="Demo of fitting a line with least squares with pytorch.")
parser.add_argument("--Nsamples", type=int, default=5000)
parser.add_argument("--CUDA", action="store_true")
args = parser.parse_args()

np.random.seed(123456)
seed_out = torch.manual_seed(123456)

dtype = torch.FloatTensor

if args.CUDA:
    torch.cuda.manual_seed(123456)
    dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
N = args.Nsamples
x = torch.linspace(-5, 5, N).type(dtype)
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties
#   and non-trivial correlated errors.
yerr = 0.1 + 0.4 * torch.rand(N).type(dtype)
yerr_hom = 0.4*torch.ones(N).type(dtype)
hom_cov = torch.diag(yerr_hom ** 2).type(dtype)
iid_cov = torch.diag(yerr ** 2).type(dtype)
true_cov = 0.5 * torch.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + torch.diag(yerr ** 2)
#y = np.random.multivariate_normal(y, true_cov)
#np.save('y_fake.npy', y)
y = np.load('y_fake.npy')
y_pt = torch.from_numpy(y).type(dtype)


#Linear algebra
A = torch.ones((N,2))
A[:, 0] = x

AT= A.t()
C = iid_cov

t0 = time.time()
C_inv = torch.inverse(C)
S_inv = torch.matmul(torch.matmul(A.t(), C_inv), A)
S = torch.inverse(S_inv)
ls_m, ls_b = torch.matmul(torch.matmul( torch.matmul(S, AT), C_inv), y_pt)
t1 = time.time()

net_time = t1-t0
print(" m: {:.2f} \n b: {:.2f} \n time: {}".format(ls_m, ls_b, net_time))
