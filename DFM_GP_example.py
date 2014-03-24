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

import emcee  # http://dan.iel.fm/emcee
import triangle  # https://github.com/dfm/triangle.py
import numpy as np
import matplotlib.pyplot as plt
from astroML.plotting import hist
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=16, usetex=False)
#from IPython.display import display, Math, Latex

np.random.seed(123456)

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
N = 50
x = 10 * np.sort(np.random.rand(N)) - 5
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties and non-trivial correlated errors.
yerr = 0.1 + 0.4 * np.random.rand(N)
true_cov = 0.5 * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + np.diag(yerr ** 2)
arg1=x[:, None]-x[None, :]
y = np.random.multivariate_normal(y, true_cov)

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

'''
Plot II: #Make an image of the noise
'''
#Visualize the covariance
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)
ax = fig.add_subplot(111)
 
ax.imshow(true_cov, interpolation="nearest", cmap="gray")
plt.savefig("figures/line-cov.pdf");
#plt.show()


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
print 1