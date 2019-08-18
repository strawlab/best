import numpy as np
import pymc3 as pm
import scipy.stats


def calculate_sample_statistics(sample_vec):
    hdi_min, hdi_max = pm.hpd(sample_vec, alpha=0.05)

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
