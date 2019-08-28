"""Make plots for displaying results of BEST test.

This module produces plots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.
"""
from typing import Optional, Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib as mpl
    mpl.rcParams['backend'] = 'TkAgg'
    import matplotlib.pyplot as plt

import numpy as np
import matplotlib.lines as mpllines
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import LogFormatter
import pymc3 as pm
import scipy.stats as st

from .model import BestResults, BestResultsOne, BestResultsTwo

# Only 99.5% of the samples are displayed, to prevent the long tails
DISPLAYED_ALPHA = 0.005
PRETTY_BLUE = '#89d1ea'


def plot_posterior(best_results: BestResults,
                   var_name: str,
                   ax: Optional[plt.Axes] = None,
                   bins: Union[int, list, np.ndarray] = 30,
                   stat: str = 'mode',
                   title: Optional[str] = None,
                   label: Optional[str] = None,
                   ref_val: Optional[float] = None,
                   **kwargs):
    """Plot a histogram of posterior samples of a variable

    Parameters
    ----------
    best_results : BestResults
        The result of the analysis.
    var_name : string
        The name of the variable to be plotted.
    ax : Matplotlib Axes, optional
        If not None, the Matplotlib Axes instance to be used.
        Default: create a new axes.
    bins : int or list or NumPy array
        The number or edges of the bins used for the histogram of the data.
        If an integer, the number of bins to use.
        If a sequence, then the edges of the bins, including left edge
        of the first bin and right edge of the last bin.
        Default: 30 bins.
    stat : {'mean', 'mode'}
        Whether to print the mean or the mode of the variable on the plot.
        Default: 'mode'.
    title : string, optional
        Title of the plot. Default: don’t print a title.
    label : string, optional
        Label of the *x* axis. Default: don’t print a label.
    ref_val : float, optional
        If not None, print a vertical line at this reference value (typically
        zero).
        Default: None (don’t print a reference value)
    **kwargs : dict
        All other keyword arguments are passed to `plt.hist`.

    Returns
    -------
    Matplotlib Axes
        The Axes object containing the plot. Using this return value, the
        plot can be customized afterwards – for details, see the documentation
        of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.

    Examples
    --------
    To plot a histogram of the samples of the mean of the first group in green
    (the hist_kwargs)

        >>> import matplotlib as plt
        >>> ax = best.plot_posterior(best_out, 'Group 1 mean', color='green')
        >>> plt.show()
    """
    samples = best_results.trace[var_name]
    samples_min, samples_max = best_results.hpd(var_name, DISPLAYED_ALPHA)
    samples = samples[(samples_min <= samples) * (samples <= samples_max)]

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    if ax is None:
        _, ax = plt.subplots()

    hist_kwargs = {'bins': bins}
    hist_kwargs.update(kwargs)
    ax.hist(samples, rwidth=0.8,
            facecolor=PRETTY_BLUE, edgecolor='none', **hist_kwargs)

    if stat:
        if stat == 'mode':
            stat_val = best_results.posterior_mode(var_name)
        elif stat == 'mean':
            stat_val = np.mean(samples)
        else:
            raise ValueError('stat parameter must be either "mean" or "mode" '
                             'or None.')

        ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                transform=trans,
                horizontalalignment='center',
                verticalalignment='top',
                )

    if ref_val is not None:
        ax.axvline(ref_val, linestyle=':')

    # plot HDI
    hdi_min, hdi_max = pm.hpd(samples, alpha=0.05)

    hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                        lw=5.0, color='k')
    hdi_line.set_clip_on(False)
    ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
            transform=trans,
            horizontalalignment='center',
            verticalalignment='bottom',
            )
    ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
            transform=trans,
            horizontalalignment='center',
            verticalalignment='bottom',
            )

    ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
            transform=trans,
            horizontalalignment='center',
            verticalalignment='bottom',
            )

    # make it pretty
    ax.spines['bottom'].set_position(('outward', 2))
    for loc in ['left', 'top', 'right']:
        ax.spines[loc].set_color('none')  # don't draw
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks([])  # don't draw
    for line in ax.get_xticklines():
        line.set_marker(mpllines.TICKDOWN)
    if label:
        ax.set_xlabel(label)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_normality_posterior(best_results, ax, bins, title):
    # TODO merge it into plot_posterior, with a log_x: bool = False parameter
    #  Then we could also center the "95% HPD" text on the log scale.

    samples = best_results.trace['Normality']
    norm_bins = np.logspace(np.log10(best_results.model.nu_min),
                            np.log10(pm.hpd(samples, alpha=DISPLAYED_ALPHA)[-1]),
                            num=bins + 1)
    plot_posterior(best_results,
                   'Normality',
                   ax=ax,
                   bins=norm_bins,
                   title=title,
                   label=r'$\nu$')
    ax.set_xlim(2.4, norm_bins[-1] * 1.05)
    ax.semilogx()
    # don't use scientific notation for tick labels
    tick_fmt = LogFormatter()
    ax.xaxis.set_major_formatter(tick_fmt)
    ax.xaxis.set_minor_formatter(tick_fmt)


def plot_data_and_prediction(best_results: BestResults,
                             group_id: int = 1,
                             ax: plt.Axes = None,
                             bins: Union[int, list, np.ndarray] = 30,
                             title: Optional[str] = None,
                             hist_kwargs: dict = {},
                             prediction_kwargs: dict = {}):
    """Plot samples of predictive distributions and a histogram of the data.

    This plot can be used as a *posterior predictive check*, to examine
    how well the model predictions fit the observed data.

    Parameters
    ----------
    best_results
        The result of the analysis.
    group_id : {1, 2}
        Which group to plot (1 or 2).
    ax : Matplotlib Axes, optional
        If not None, the Matplotlib Axes instance to be used.
        Default: create a new plot.
    title : string, optional.
        Title of the plot. Default: no plot title.
    bins : int or list or NumPy array.
        The number or edges of the bins used for the histogram of the data.
        If an integer, the number of bins to use.
        If a sequence, then the edges of the bins, including left edge
        of the first bin and right edge of the last bin.
        Default: 30 bins.
    hist_kwargs : dict
        The keyword arguments to be passed to `plt.hist` for the group data.
    prediction_kwargs : dict
        The keyword arguments to be passed to `plt.plot` for the posterior
        predictive curves.

    Returns
    -------
    Matplotlib Axes
        The Axes object containing the plot. Using this return value, the
        plot can be customized afterwards – for details, see the documentation
        of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.

    Examples
    --------
    To print the data of the second group, add a hatch to the histogram, and
    set the limits of the *x* axis to 85 and 115:

        >>> import matplotlib as plt
        >>> ax = best.plot_data_and_prediction(
        ...         best_out,
        ...         2,
        ...         hist_kwargs={'hatch':'...'}
        ... )
        >>> ax.set_xlim(85, 115)
        >>> plt.show()

    Notes
    -----
    You can move the histogram in front of the predictive curves by passing
    ``hist_kwargs={'zorder': 10}`` as an argument, or completely behind the
    curves with ``hist_kwargs={'zorder': 0}``.

    If the plot is large enough, it is suggested to put a legend on it, by
    calling ``ax.legend()`` afterwards.
    """

    if ax is None:
        _, ax = plt.subplots()

    group_data = best_results.observed_data(group_id)
    trace = best_results.trace

    if isinstance(best_results, BestResultsTwo):
        means = trace['Group %d mean' % group_id]
        sigmas = trace['Group %d sigma' % group_id]
        nus = trace['Normality']
    elif isinstance(best_results, BestResultsOne):
        means = trace['Mean']
        sigmas = trace['Sigma']
        nus = trace['Normality']
    else:
        raise ValueError('Unknown type of best_results argument')

    n_curves = 50
    n_samps = len(means)
    idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)

    try:
        xmin = bins[0]
        xmax = bins[-1]
    except TypeError:
        xmin = np.min(group_data)
        xmax = np.max(group_data)

    dx = xmax - xmin
    xmin -= dx * 0.05
    xmax += dx * 0.05

    x = np.linspace(xmin, xmax, 1000)

    kwargs = dict(color=PRETTY_BLUE, zorder=1, alpha=0.3)
    kwargs.update(prediction_kwargs)

    for i in idxs:
        v = st.t.pdf(x, nus[i], means[i], sigmas[i])
        line, = ax.plot(x, v, **kwargs)

    line.set_label('Prediction')

    kwargs = dict(edgecolor='w',
                  facecolor='xkcd:salmon',
                  density=True,
                  bins=bins,
                  label='Observation')
    kwargs.update(hist_kwargs)
    ax.hist(group_data, **kwargs)

    # draw a translucent histogram in front of the curves
    if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
        kwargs.update(dict(zorder=3, label=None, alpha=0.3))
        ax.hist(group_data, **kwargs)

    ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top'
            )

    for loc in ['top', 'right']:
        ax.spines[loc].set_color('none')  # don't draw
    ax.spines['left'].set_color('gray')
    ax.set_xlabel('Observation')
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('Probability')
    ax.set_yticks([])
    ax.set_ylim(0)
    if title:
        ax.set_title(title)

    return ax


def plot_all_two(best_results: BestResultsTwo,
                 bins: int = 30,
                 group1_name: str = 'Group 1',
                 group2_name: str = 'Group 2') -> plt.Figure:
    """Plot posteriors of every parameter and observation of a two-group analysis.

    Parameters
    ----------
    best_results : BestResultsTwo
        The result of the analysis.
    bins : int
        The number of bins to be used for the histograms.
        Default: 30.
    group1_name : string
        Name of the first group, to be used in the titles.
        Default: "Group 1".
    group2_name : string
        Name of the second group, to be used in the titles.
        Default: "Group 2".

    Returns
    -------
    plt.Figure
        The created figure. (The separate plots can be accessed via
        ``fig.axes``, where ``fig`` is the return value of this function.)
    """
    assert type(bins) is int, 'bins argument must be an integer.'

    trace = best_results.trace

    posterior_mean1 = trace['Group 1 mean']
    posterior_mean2 = trace['Group 2 mean']

    posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
    _, bin_edges_means = np.histogram(posterior_means, bins=bins)

    posterior_std1 = trace['Group 1 SD']
    posterior_std2 = trace['Group 2 SD']

    std1_min, std1_max = best_results.hpd('Group 1 SD', DISPLAYED_ALPHA)
    std2_min, std2_max = best_results.hpd('Group 2 SD', DISPLAYED_ALPHA)
    std_min = min(std1_min, std2_min)
    std_max = max(std1_max, std2_max)
    stds = np.concatenate((posterior_std1, posterior_std2))
    stds = stds[(std_min <= stds) * (stds <= std_max)]
    _, bin_edges_stds = np.histogram(stds, bins=bins)

    fig, axes = plt.subplots(5, 2, figsize=(8.2, 11))

    axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])

    plot_posterior(best_results,
                   'Group 1 mean',
                   ax=axes[0, 0],
                   bins=bin_edges_means,
                   stat='mean',
                   title='%s mean' % group1_name,
                   label=r'$\mu_1$')

    plot_posterior(best_results,
                   'Group 2 mean',
                   ax=axes[1, 0],
                   bins=bin_edges_means,
                   stat='mean',
                   title='%s mean' % group2_name,
                   label=r'$\mu_2$')

    axes[2, 0].get_shared_x_axes().join(axes[2, 0], axes[3, 0])

    plot_posterior(best_results,
                   'Group 1 SD',
                   ax=axes[2, 0],
                   bins=bin_edges_stds,
                   title='%s std. dev.' % group1_name,
                   label=r'$\mathrm{sd}_1$')

    plot_posterior(best_results,
                   'Group 2 SD',
                   ax=axes[3, 0],
                   bins=bin_edges_stds,
                   title='%s std. dev.' % group2_name,
                   label=r'$\mathrm{sd}_2$')

    plot_normality_posterior(best_results, axes[4, 0], bins, 'Normality')

    plot_posterior(best_results,
                   'Difference of means',
                   ax=axes[2, 1],
                   bins=bins,
                   title='Difference of means',
                   stat='mean',
                   ref_val=0,
                   label=r'$\mu_1 - \mu_2$')

    plot_posterior(best_results,
                   'Difference of SDs',
                   ax=axes[3, 1],
                   bins=bins,
                   title='Difference of std. dev.s',
                   ref_val=0,
                   label=r'$\mathrm{sd}_1 - \mathrm{sd}_2$')

    plot_posterior(best_results,
                   'Effect size',
                   ax=axes[4, 1],
                   bins=bins,
                   title='Effect size',
                   ref_val=0,
                   label=r'$(\mu_1 - \mu_2) /'
                          r' \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2)/2}$')

    group1_data = best_results.observed_data(1)
    group2_data = best_results.observed_data(2)
    obs_vals = np.concatenate((group1_data, group2_data))
    bin_edges = np.linspace(np.min(obs_vals), np.max(obs_vals), bins + 1)

    axes[0, 1].get_shared_x_axes().join(axes[0, 1], axes[1, 1])

    plot_data_and_prediction(best_results, 1,
                             ax=axes[0, 1],
                             bins=bin_edges,
                             title='%s data with post. pred.' % group1_name)

    plot_data_and_prediction(best_results, 2,
                             ax=axes[1, 1],
                             bins=bin_edges,
                             title='%s data with post. pred.' % group2_name)

    fig.tight_layout()

    return fig


def plot_all_one(best_results: BestResultsOne,
                 bins: int = 30,
                 group_name: Optional[str] = None) -> plt.Figure:
    """Plot posteriors of every parameter and observation of a two-group analysis.

    Parameters
    ----------
    best_results : BestResultsOne
        The result of the analysis.
    bins : int
        The number of bins to be used for the histograms.
        Default: 30.
    group_name : string, optional
        If not None, group name to be used in the title, e.g. if
         ``group_name`` is ``"eTRF day 5"`` then the plot for the mean is titled
         “eTRF day 5 mean”.
        If None, then group name is omitted from the titles, resulting in
         e.g. “Mean”.
        Default: None.

    Returns
    -------
    plt.Figure
        The created figure. (The separate plots can be accessed via
        ``fig.axes``, where ``fig`` is the return value of this function.)
    """
    assert type(bins) is int, 'bins argument must be an integer.'

    def maybe_caps(title):
        if group_name:
            return group_name + ' ' + title
        else:
            return title.capitalize()

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 6.6))

    plot_posterior(best_results,
                   'Mean',
                   ax=axes[0, 0],
                   bins=bins,
                   stat='mean',
                   title=maybe_caps('mean'),
                   label=r'$\mu$')

    plot_posterior(best_results,
                   'SD',
                   ax=axes[0, 1],
                   bins=bins,
                   title=maybe_caps('std. dev.'),
                   label=r'$\sigma$')

    plot_normality_posterior(best_results,
                             axes[1, 0],
                             bins,
                             maybe_caps('normality'))

    ref_val = best_results.model.ref_val
    if ref_val == 0:
        label = r'$\mu / \sigma$'
    else:
        label = r'$(\mu - %.1f) / \sigma$' % ref_val

    plot_posterior(best_results,
                   'Effect size',
                   ax=axes[1, 1],
                   bins=bins,
                   title=maybe_caps('effect size'),
                   ref_val=ref_val,
                   label=label)

    plot_data_and_prediction(best_results, 1,
                             ax=axes[2, 0],
                             bins=bins,
                             title=maybe_caps('data with post. pred.'))

    fig.delaxes(axes[2, 1])
    fig.tight_layout()

    return fig


def plot_all(best_results: BestResults,
             *args,
             **kwargs) -> plt.Figure:
    """Plot posteriors of every parameter and observation of an analysis.

    Depending on the type of best_results, this call is equivalent to calling
    :func:`plot_all_one` or :func:`plot_all_two`.
    """
    if isinstance(best_results, BestResultsOne):
        return plot_all_one(best_results, *args, **kwargs)
    elif isinstance(best_results, BestResultsTwo):
        return plot_all_two(best_results, *args, **kwargs)
    else:
        raise ValueError('best_results argument is of unknown type')
