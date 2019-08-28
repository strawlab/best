"""Bayesian estimation for two groups

This module implements Bayesian estimation for two groups, providing
complete distributions for effect size, group means and their
difference, standard deviations and their difference, and the
normality of the data.

Based on:

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.

"""

import sys

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
        # Note: the IDE might give a warning for these because it thinks
        #  distributions like pm.Normal() don't have a string "name" argument,
        #  but this is false – pm.Distribution redefined __new__, so the
        #  first argument indeed is the name (a string).
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


def make_model_one_group(y, ref_val=0):
    y = np.array(y)

    assert y.ndim == 1

    mu_m = np.mean(y)
    mu_scale = np.std(y) * 1000

    sigma_low = np.std(y) / 1000
    sigma_high = np.std(y) * 1000

    with pm.Model() as model:
        # the five prior distributions for the parameters in our model
        # Note: the IDE might give a warning for these because it thinks
        #  distributions like pm.Normal() don't have a string "name" argument,
        #  but this is false – pm.Distribution redefined __new__, so the
        #  first argument indeed is the name (a string).
        group_mean = pm.Normal('Mean', mu=mu_m, sd=mu_scale)
        group_std = pm.Uniform('SD', lower=sigma_low, upper=sigma_high)
        group_prec = group_std ** (-2)
        nu = pm.Exponential('nu - 1', 1 / 29.) + 1
        _ = pm.Deterministic('Normality', np.log10(nu))
        _ = pm.StudentT('Data', observed=y, nu=nu, mu=group_mean, lam=group_prec)
        _ = pm.Deterministic('Effect size', (group_mean - ref_val) / group_std)

    return model


def analyze_two(group1_data,
                group2_data,
                n_samples: int = 2000,
                **kwargs):
    """Analyze the difference between two groups

    This analysis takes about a minute, depending on the amount of data.
    (See the Notes section below.)

    This function creates a model with the given parameters, and updates the
    distributions of the parameters as dictated by the model and the data.

    Parameters
    ----------
    group1_data : list of numbers, NumPy array, or Pandas Series.
        Data of the first group analyzed and to be plotted.
    group2_data : list of numbers, NumPy array, or Pandas Series.
        Data of the second group analyzed and to be plotted.
    n_samples : int
        Number of samples *per chain* to be drawn for the analysis.
        (The number of chains depends on the number of CPU cores, but is
        at least 2.) Default: 2000.
    **kwargs
        Keyword arguments are passed to :meth:`pymc3.sample`.
        For example, number of tuning samples can be increased to 2000
        (from the default 1000) by::

            best.analyze_two(group1_data, group2_data, tune=2000)

    Notes
    -----
    The first call of this function takes 2 minutes extra, in order to
    compile the model and speed up later calls.

    Afterwards, performing a two-group analysis takes:
     - 20 seconds with 45 data points per group, or
     - 90 seconds with 10,000 data points per group.

    Don’t be intimidated by the time estimates in the beginning – the sampling
    process speeds up after the initial few hundred iterations.

    (These times were measured on a 2015 MacBook.)
    """
    model = make_model_two_groups(group1_data, group2_data)
    trace = perform_sampling(model, n_samples, kwargs)

    return trace


def analyze_one(group_data,
                ref_val: float = 0,
                n_samples: int = 2000,
                **kwargs):
    """Analyze the distribution of a single group

    This method is typically used to compare some observations against a
    reference value, such as zero. It can be used to analyze paired data,
    or data from an experiment without a control group.

    This analysis takes around a minute, depending on the amount of data.

    This function creates a model with the given parameters, and updates the
    distributions of the parameters as dictated by the model and the data.

    Parameters
    ----------
    group_data : list of numbers, NumPy array, or Pandas Series.
        Data of the group to be analyzed.
    ref_val : float
        The reference value to be compared against. Default: 0.
    n_samples : int
        Number of samples *per chain* to be drawn for the analysis.
        (The number of chains depends on the number of CPU cores, but is
        at least 2.) Default: 2000.
    **kwargs
        Keyword arguments are passed to :meth:`pymc3.sample`.
        For example, number of tuning samples can be increased to 2000
        (from the default 1000) by::

            best.analyze_one(group_data, tune=2000)
    """
    model = make_model_one_group(group_data, ref_val)
    trace = perform_sampling(model, n_samples, kwargs)

    return trace


def perform_sampling(model, n_samples, kwargs):
    kwargs['tune'] = kwargs.get('tune', 1000)
    kwargs['nuts_kwargs'] = kwargs.get('nuts_kwargs', {'target_accept': 0.90})
    for _ in range(2):
        with model:
            trace = pm.sample(n_samples, **kwargs)

        if trace.report.ok:
            break
        else:
            kwargs['tune'] = 2000
            print("\nDue to potentially incorrect estimates, rerunning sampling "
                  "with {} tuning samples.\n".format(kwargs['tune']), file=sys.stderr)
    return trace


# We make an alias for pm.summary() too so that the user doesn't need to import
#  the `pymc3` module explicitly.
summary = pm.summary
