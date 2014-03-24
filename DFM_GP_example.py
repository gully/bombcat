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
import matplotlib.pyplot as pl
from IPython.display import display, Math, Latex

np.random.seed(123456)

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
N = 50
x = 10 * np.sort(np.random.rand(N)) - 5
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties and non-trivial correlated errors.
yerr = 0.1 + 0.4 * np.random.rand(N)
true_cov = 0.5 * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + np.diag(yerr ** 2)
y = np.random.multivariate_normal(y, true_cov)


#And plot the data with the observational uncertainties. The true line is plotted in black.
x0 = np.linspace(-6, 6, 1000)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.plot(x0, true_m * x0 + true_b, "k", lw=2, alpha=0.8)
pl.ylim(-4, 4)
pl.savefig("figures/line-data.pdf");

print 1