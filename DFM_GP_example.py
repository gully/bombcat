# -*- coding: utf-8 -*-
"""

Author: gully
Date:   Mar 24 16:35:28 2014
Desc:   Gaussian Process example

Originals available at:
https://github.com/dfm/gp-tutorial
&
https://speakerdeck.com/dfm/an-astronomers-introduction-to-gaussian-processes

"""

from matplotlib import rcParams
rcParams["savefig.dpi"] = 150
import prettyplotlib as ppl
import emcee  # http://dan.iel.fm/emcee
import triangle  # https://github.com/dfm/triangle.py
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=16, usetex=False)
#from IPython.display import display, Math, Latex

#np.random.seed(123456)
np.random.seed(78704)

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
N = 50
x = 10 * np.sort(np.random.rand(N)) - 5
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties 
#   and non-trivial correlated errors.
yerr = 0.1 + 0.4 * np.random.rand(N)
iid_cov = np.diag(yerr ** 2)
true_cov = 0.5 * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + np.diag(yerr ** 2)
y = np.random.multivariate_normal(y, true_cov)
#y = np.random.multivariate_normal(y, iid_cov)

'''
Plot I: #Make a histogram of the noise
'''
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)
ax = fig.add_subplot(111)
hist(yerr, bins='knuth', ax=ax, normed=True, histtype='stepfilled', alpha=0.4)
ax.set_xlabel('$y_{err}$')
ax.set_ylabel('$p(y_{err})$')
plt.savefig("figures/yerr.pdf");
plt.close()
'''
Plot II: #Make an image of the noise
'''
#Visualize the covariance
fig, ax = plt.subplots(1)
ppl.pcolormesh(fig, ax, true_cov)
fig.savefig('figures/pplLineCov.png')
#plt.show()
plt.close()
'''
Plot III: #Data vs 'truth'
''' 
#And plot the data with the observational uncertainties. The true line is plotted in black.
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)

ax = fig.add_subplot(111)
x0 = np.linspace(-6, 6, 1000)
ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, true_m * x0 + true_b, "k", lw=2, alpha=0.8)
ax.set_ylim(-4, 4)
plt.savefig("figures/line-data.pdf");
#plt.show()
plt.close()

'''
Do the linear regression in Matrix form!
'''
A = np.vander(x, 2)
AT= A.T
C = iid_cov
C_inv = np.linalg.inv(C)
S_inv = np.dot( np.dot(AT, C_inv), A)
S= np.linalg.inv(S_inv)

ls_m, ls_b = np.linalg.solve(S_inv, np.dot(A.T, np.linalg.solve(C, y)))
ls_S = np.linalg.inv(Sinv)

'''
Visualize the Least Squares parameters as a Hess diagram
'''
#Pick m, b at random
rand_params=np.random.multivariate_normal([ls_m, ls_b], ls_S, size=5000)

H, xbins, ybins = np.histogram2d(rand_params[:,0], rand_params[:,1],
                                 bins=(np.linspace(0.33, 0.42, 50),
                                       np.linspace(0.0, 0.29, 50)))

# Create a black and white color map where bad data (NaNs) are white
cmap = plt.cm.binary
cmap.set_bad('w', 1.)

# Use the image display function imshow() to plot the result
fig, ax = plt.subplots(figsize=(8, 6))
H[H == 0] = 1  # prevent warnings in log10
ax.imshow(np.log10(H).T, origin='lower',
          extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
          cmap=cmap, interpolation='nearest',
          aspect='auto')

ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$b$')

ax.set_xlim(0.33, 0.42)
ax.set_ylim(0.0, 0.29)

plt.savefig("figures/leastSqParDist.pdf")
plt.close()

'''
Plot the least squares solution as an oband
'''
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)
ax = fig.add_subplot(111)

samples = np.dot(np.vander(x0, 2), rand_params.T)
ls_mu = np.mean(samples, axis=1)
ls_std = np.std(samples, axis=1)

ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, true_m * x0 + true_b, "k", lw=2, alpha=0.8);
ax.fill_between(x0, ls_mu+ls_std, ls_mu-ls_std, color="r", alpha=0.3)
ax.set_ylim(-4, 4)
plt.savefig("figures/line-ls.pdf");
plt.close()


print 1