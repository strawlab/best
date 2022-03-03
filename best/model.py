"""Bayesian estimation for two groups

This module implements Bayesian estimation for two groups, providing
complete distributions for effect size, group means and their
difference, standard deviations and their difference, and the
normality of the data.

Based on:

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.

"""

import numpy as np
import pymc3 as pm


def make_model(y1, y2):
    y1 = np.array(y1)
    y2 = np.array(y2)

    assert y1.ndim == 1
    assert y2.ndim == 1

    y_all = np.concatenate((y1, y2))

    mu_m = np.mean(y_all)
    mu_scale = np.std(y_all) * 1000

    sigma_low = np.std(y_all) / 1000
    sigma_high = np.std(y_all) * 1000

    with pm.Model() as model:
        # the five prior distributions for the parameters in our model
        group1_mean = pm.Normal('Group 1 mean', mu=mu_m, sd=mu_scale)
        group2_mean = pm.Normal('Group 2 mean', mu=mu_m, sd=mu_scale)
        group1_std = pm.Uniform('Group 1 SD', lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform('Group 2 SD', lower=sigma_low, upper=sigma_high)
        lambda1 = group1_std ** (-2)
        lambda2 = group2_std ** (-2)
        nu = pm.Exponential('nu - 1', 1 / 29.) + 1
        normality = pm.Deterministic('Normality', nu)

        group1_obs = pm.StudentT('Group 1 data', observed=y1, nu=nu, mu=group1_mean, lam=lambda1)
        group2_obs = pm.StudentT('Group 2 data', observed=y2, nu=nu, mu=group2_mean, lam=lambda2)

        diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic('Difference of SDs', group1_std - group2_std)
        effect_size = pm.Deterministic('Effect size', diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2))

    return model
