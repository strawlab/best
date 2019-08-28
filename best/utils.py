import numpy as np
import pymc3 as pm
from pymc3.backends.base import MultiTrace
import scipy.stats


def posterior_mode(best_out: MultiTrace,
                   variable_name: str):
    """Calculate the posterior mode of a variable

    Parameters
    ----------
    best_out : PyMC3 MultiTrace
        The result of the analysis.
    variable_name : string
        The name of the variable whose posterior mode is to be calculated.

    Returns
    -------
    float
        The posterior mode.
    """
    sample_vec = best_out.get_values(variable_name)

    # calculate mode using kernel density estimate
    kernel = scipy.stats.gaussian_kde(sample_vec)

    bw = kernel.covariance_factor()
    cut = 3 * bw
    xlow = np.min(sample_vec) - cut * bw
    xhigh = np.max(sample_vec) + cut * bw
    n = 512
    x = np.linspace(xlow, xhigh, n)
    vals = kernel.evaluate(x)
    max_idx = np.argmax(vals)
    mode_val = x[max_idx]

    return mode_val


# We make an alias for these functions so that the user doesn't need to import
#  the `pymc3` module explicitly.
summary = pm.summary
hpd = pm.hpd
