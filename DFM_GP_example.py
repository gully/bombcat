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
from astroML.stats.random import bivariate_normal
from matplotlib.patches import Ellipse
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

ls_m, ls_b = np.linalg.solve(S_inv, np.dot(A.T, np.linalg.solve(iid_cov, y)))
ls_S = np.linalg.inv(S_inv)


'''
Plot the least squares solution as an oband
'''
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)
ax = fig.add_subplot(111)

rand_params=np.random.multivariate_normal([ls_m, ls_b], ls_S, size=5000)
samples = np.dot(np.vander(x0, 2), rand_params.T)
ls_mu = np.mean(samples, axis=1)
ls_std = np.std(samples, axis=1)

ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, true_m * x0 + true_b, "k", lw=2, alpha=0.8);
ax.fill_between(x0, ls_mu+ls_std, ls_mu-ls_std, color="r", alpha=0.3)
ax.set_ylim(-4, 4)
plt.savefig("figures/line-ls.pdf");
plt.close()


'''
Okay, now let's use the True Covariance matrix we have and do the same least squares.
This is sort of cheating, since the covariance matrix was used in drawing the data
in the first place.  It's important to note that we don't get back exactly the
true line when we stick in the covariance matrix though.  The y's were merely DRAWN FROM 
a multivariate normal (MVN) whose mean values were the true y's (=true_m x + true_b),
and the covariance matrix of the MVN was this True Covariance matrix, true_cov.
In the limit that we fit tons of iterations of the least squares, one of the fits
will be close to true_m, true_b, but there will be ample (honest) scatter.
'''
S_inv1 = np.dot(AT, np.linalg.solve(true_cov, A))
corr_m, corr_b = np.linalg.solve(S_inv1, np.dot(AT, np.linalg.solve(true_cov, y)))
corr_S = np.linalg.inv(S_inv1)

rand_paramsCorr=np.random.multivariate_normal([corr_m, corr_b], corr_S, size=5000)

'''
Visualize the [m, b] derived from independent error Least Squares AND 
    true Correlation matrix least squares, and the true [m, b]
    Put the distribution of parameters as a Hess diagram
    Really, we could skip this and just plot the contours
'''
#Pick m, b at random

H, xbins, ybins = np.histogram2d(rand_params[:,0], rand_params[:,1],
                                 bins=(np.linspace(0.33, 0.42, 50),
                                       np.linspace(0.0, 0.29, 50)))
HC, xbinsC, ybinsC = np.histogram2d(rand_paramsCorr[:,0], rand_paramsCorr[:,1],
                                 bins=(np.linspace(0.25,0.75, 25),
                                       np.linspace(-0.5,0.5, 25)))
# Create a black and white color map where bad data (NaNs) are white
cmap = plt.cm.binary
cmap.set_bad('w', 1.)

# Use the image display function imshow() to plot the result
fig, ax = plt.subplots(figsize=(8, 6))
H[H == 0] = 1  # prevent warnings in log10
HC[HC == 0] = 1  # prevent warnings in log10
'''
ax.imshow(np.log10(H).T, origin='lower',
          extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
          cmap=cmap, interpolation='nearest',
          aspect='auto')
'''
ax.imshow(np.log10(HC).T, origin='lower',
          extent=[xbinsC[0], xbinsC[-1], ybinsC[0], ybinsC[-1]],
          cmap=cmap, interpolation='nearest',
          aspect='auto')          


ax.plot([ls_m], [ls_b], 'rx')  
ax.plot([corr_m], [corr_b], 'b+')      
ax.plot([true_m], [true_b], 'go')  

#From astroML book page 110 eq. 3.82
ang1=0.5*np.arctan(2.0*ls_S[1,0]/(ls_S[0,0]-ls_S[1,1]))
ang2=0.5*np.arctan(2.0*corr_S[1,0]/(corr_S[0,0]-corr_S[1,1]))

for N in (1, 2, 3):
    ax.add_patch(Ellipse([corr_m, corr_b], N * sqrt(corr_S[0,0]), N*sqrt(corr_S[1,1]),
                         angle=ang2 * 180. / np.pi, lw=1,
                         ec='b', fc='none'))

for N in (1, 2, 3):
    ax.add_patch(Ellipse(np.array([ls_m, ls_b]), N * sqrt(ls_S[0,0]), N * sqrt(ls_S[1,1]),
                         angle=ang1 * 180. / np.pi, lw=1,
                         ec='r', fc='none'))

ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$b$')

ax.set_xlim(0.25, 0.75)
ax.set_ylim(-0.5, 0.5)

plt.savefig("figures/leastSqParDist.pdf")
plt.close()


'''
Plot a nice big blue band over the data- the band is the uncertainty
surrounding a line fit, for a range of m's and b's drawn from the MVN centered on
the linear regression with the true covariance matrix 
'''

samples = np.dot(np.vander(x0, 2), rand_paramsCorr.T)
corr_mu = np.mean(samples, axis=1)
corr_std = np.std(samples, axis=1)

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.3, bottom=0.17, top=0.9)
ax = fig.add_subplot(111)

ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, true_m * x0 + true_b, "k", lw=2, alpha=0.8);
ax.fill_between(x0, corr_mu+corr_std, corr_mu-corr_std, color="b", alpha=0.3)
ax.fill_between(x0, ls_mu+ls_std, ls_mu-ls_std, color="r", alpha=0.3)
plt.ylim(-4, 4)
plt.savefig("figures/line-true_cov.pdf");
plt.close()

'''
Now we move on to applying a Kernel to model the off-diagonal elements
for an unknown covariance matrix!
To do that we should define some convenient functions
'''
#"Log Likelihood"
#Note: x is not passed to this function.  It must be a global variable...
def lnlike(m, b, lna, lns):
    a, s = np.exp(lna), np.exp(lns)
    C = np.diag(yerr**2) + a * np.exp(-0.5 * (x[:, None] - x[None, :])**2 / s)
    s, logdet = np.linalg.slogdet(C)
    if s <= 0:
        return -np.inf
    r = y - (m*x + b)
    return -0.5 * (np.dot(r, np.linalg.solve(C, r)) + logdet)

# Apply a uniform prior over some range.
# Shape params are uniform in log space 
def lnprior(m, b, lna, lns):
    if not (-2 < m < 2 and -2 < b < 2 and -5 < lna < 5 and -5 < lns < 5):
        return -np.inf
    return 0.0

#I'm honestly not sure the point of this function... what are these *p's?
#This basically just adds zero (prior) to the likelihood, or is -inf
def lnprob(p):
    lp = lnprior(*p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(*p)


print 1