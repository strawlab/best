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


def make_model_two_groups(y1, y2):
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
        _ = pm.Deterministic('Normality', nu)

        _ = pm.StudentT('Group 1 data', observed=y1, nu=nu, mu=group1_mean, lam=lambda1)
        _ = pm.StudentT('Group 2 data', observed=y2, nu=nu, mu=group2_mean, lam=lambda2)

        diff_of_means = pm.Deterministic('Difference of means', group1_mean - group2_mean)
        _ = pm.Deterministic('Difference of SDs', group1_std - group2_std)
        _ = pm.Deterministic('Effect size', diff_of_means / np.sqrt((group1_std ** 2 + group2_std ** 2) / 2))

    return model


def analyze_two(y1, y2, n_samples=2000, **kwargs):
    """Analyze the difference between two groups

    This function creates a model with the given parameters, and updates the
    distributions of the parameters as dictated by the model and the data.

    :param **kwargs: Keyword arguments are passed to :meth:`pymc3.sample`.
        For example, number of tuning samples can be increased to 2000
        (from the default 1000) by::

            best.analyze_two(group_1_data, group_2_data, tune=2000)

    """
    model = make_model_two_groups(y1, y2)
    kwargs['tune'] = kwargs.get('tune', 1000)
    kwargs['nuts_kwargs'] = kwargs.get('nuts_kwargs', {'target_accept': 0.90})

    with model:
        trace = pm.sample(n_samples, **kwargs)

    return trace


# We make an alias for pm.summary() too so that the user doesn't need to import
#  the `pymc3` module explicitly.
summary = pm.summary
