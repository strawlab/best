"""Bayesian estimation for two groups

This module implements Bayesian estimation for two groups, providing
complete distributions for effect size, group means and their
difference, standard deviations and their difference, and the
normality of the data.

Based on:

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.

"""

from __future__ import division
from pymc import Uniform, Normal, Exponential, NoncentralT, deterministic, Model
import numpy as np
import scipy.stats


def make_model(data):
    assert len(data) == 2, 'There must be exactly two data arrays'

    name1, name2 = sorted(data.keys())

    y1 = np.array(data[name1])
    y2 = np.array(data[name2])

    assert y1.ndim == 1
    assert y2.ndim == 1
    y = np.concatenate((y1, y2))

    mu_m = np.mean(y)
    mu_p = 0.000001 * 1 / np.std(y) ** 2

    sigma_low = np.std(y) / 1000
    sigma_high = np.std(y) * 1000

    # the five prior distributions for the parameters in our model
    group1_mean = Normal('group1_mean', mu_m, mu_p)
    group2_mean = Normal('group2_mean', mu_m, mu_p)
    group1_std = Uniform('group1_std', sigma_low, sigma_high)
    group2_std = Uniform('group2_std', sigma_low, sigma_high)
    nu_minus_one = Exponential('nu_minus_one', 1 / 29)

    @deterministic(plot=False)
    def nu(n=nu_minus_one):
        out = n + 1
        return out

    @deterministic(plot=False)
    def lam1(s=group1_std):
        out = 1 / s ** 2
        return out

    @deterministic(plot=False)
    def lam2(s=group2_std):
        out = 1 / s ** 2
        return out

    group1 = NoncentralT(name1, group1_mean, lam1, nu, value=y1, observed=True)
    group2 = NoncentralT(name2, group2_mean, lam2, nu, value=y2, observed=True)
    return Model({'group1': group1,
                  'group2': group2,
                  'group1_mean': group1_mean,
                  'group2_mean': group2_mean,
                  })


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]
    return hdi_min, hdi_max


def calculate_sample_statistics(sample_vec):
    hdi_min, hdi_max = hdi_of_mcmc(sample_vec)

    # calculate mean
    mean_val = np.mean(sample_vec)

    # calculate mode (use kernel density estimate)
    kernel = scipy.stats.gaussian_kde(sample_vec)
    if 1:
        # (Could we use the mean shift algorithm instead of this?)
        bw = kernel.covariance_factor()
        cut = 3 * bw
        xlow = np.min(sample_vec) - cut * bw
        xhigh = np.max(sample_vec) + cut * bw
        n = 512
        x = np.linspace(xlow, xhigh, n)
        vals = kernel.evaluate(x)
        max_idx = np.argmax(vals)
        mode_val = x[max_idx]
    return {'hdi_min': hdi_min,
            'hdi_max': hdi_max,
            'mean': mean_val,
            'mode': mode_val,
            }
