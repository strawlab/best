"""Make plots for displaying results of BEST test.

This module produces plots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.
"""
from __future__ import division
import numpy as np
from best import hdi_of_mcmc

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mpllines
import matplotlib.ticker as mticker
import scipy.stats
from pymc.distributions import noncentral_t_like

pretty_blue = '#89d1ea'

def plot_posterior( sample_vec, bins=None, ax=None, title=None, stat='mode',
                        label='', draw_zero=False ):

    hdi_min, hdi_max = hdi_of_mcmc( sample_vec )

    # calculate mean
    mean_val = np.mean(sample_vec)

    # calculate mode (use kernel density estimate)
    kernel = scipy.stats.gaussian_kde( sample_vec )
    if 1:
        # (Could we use the mean shift algorithm instead of this?)
        bw = kernel.covariance_factor()
        cut = 3*bw
        xlow = np.min(sample_vec) - cut*bw
        xhigh = np.max(sample_vec) + cut*bw
        n = 512
        x = np.linspace(xlow,xhigh,n)
        vals = kernel.evaluate(x)
        max_idx = np.argmax(vals)
        mode_val = x[max_idx]

    if stat=='mean':
        stat_val = mean_val
    elif stat=='mode':
        stat_val = mode_val
    else:
        raise ValueError('unknown stat %s'%stat)

    if ax is not None:
        ax.hist( sample_vec, bins=bins, rwidth=0.8,
                 facecolor=pretty_blue, edgecolor='none' )
        if title is not None:
            ax.set_title( title )

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text( stat_val, 0.99, '%s = %.3g'%(stat,stat_val),
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='top',
                 )
        if draw_zero:
            ax.axvline(0,linestyle=':')

        # plot HDI
        hdi_line, = ax.plot( [hdi_min, hdi_max], [0,0],
                             lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text( hdi_min, 0.04, '%.3g'%hdi_min,
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )
        ax.text( hdi_max, 0.04, '%.3g'%hdi_max,
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )

        ax.text( (hdi_min+hdi_max)/2, 0.14, '95% HDI',
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )

        # make it pretty
        ax.spines['bottom'].set_position(('outward',2))
        for loc in ['left','top','right']:
            ax.spines[loc].set_color('none') # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([]) # don't draw
        ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        ax.set_xlabel(label)

def plot_data_and_prediction( data, means, stds, numos, ax=None, bins=None,
                              n_curves=50, group='x'):
    assert ax is not None

    ax.hist( data, bins=bins, rwidth=0.5,
             facecolor='r', edgecolor='none', normed=True )

    if bins is not None:
        if hasattr(bins,'__len__'):
            xmin = bins[0]
            xmax = bins[-1]
        else:
            xmin = np.min(data)
            xmax = np.max(data)

    n_samps = len(means)
    idxs = map(int,np.round( np.random.uniform(size=n_curves)*n_samps ))

    x = np.linspace(xmin, xmax, 100)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y)')

    for i in idxs:
        m = means[i]
        s = stds[i]
        lam = 1/s**2
        numo = numos[i]
        nu = numo+1

        v = np.exp([noncentral_t_like(xi,m,lam,nu) for xi in x])
        ax.plot(x,v, color=pretty_blue, zorder=-10)

    ax.text(0.8,0.95,'$\mathrm{N}_{%s}=%d$'%( group, len(data), ),
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top'
            )
    ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
    ax.yaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
    ax.set_title('Data Group %s w. Post. Pred.'%(group,))

def make_figure(M):
    # plotting stuff
    n_bins = 30

    posterior_mean1 = M.trace('group1_mean')[:]
    posterior_mean2 = M.trace('group2_mean')[:]
    diff_means = posterior_mean1 - posterior_mean2

    posterior_means = np.concatenate( (posterior_mean1,posterior_mean2) )
    _, bin_edges_means = np.histogram( posterior_means, bins=n_bins )

    posterior_std1 = M.trace('group1_std')[:]
    posterior_std2 = M.trace('group2_std')[:]
    diff_stds = posterior_std1 - posterior_std2

    posterior_stds = np.concatenate( (posterior_std1,posterior_std2) )
    _, bin_edges_stds = np.histogram( posterior_stds, bins=n_bins )

    effect_size = diff_means / np.sqrt((posterior_std1**2+posterior_std2**2)/2)

    post_nu_minus_one = M.trace('nu_minus_one')[:]
    lognup = np.log10(post_nu_minus_one+1)

    f = plt.figure(figsize=(8.2,11),facecolor='white')
    ax1 = f.add_subplot(5,2,1,axisbg='none')
    plot_posterior( posterior_mean1, bins=bin_edges_means, ax=ax1,
                    title='Group 1 Mean', stat='mean',
                    label=r'$\mu_1$')

    ax3 = f.add_subplot(5,2,3,axisbg='none')
    plot_posterior( posterior_mean2, bins=bin_edges_means, ax=ax3,
                    title='Group 2 Mean', stat='mean',
                    label=r'$\mu_2$')

    ax5 = f.add_subplot(5,2,5,axisbg='none')
    plot_posterior( posterior_std1, bins=bin_edges_stds, ax=ax5,
                    title='Group 1 Std. Dev.',
                    label=r'$\sigma_1$')


    ax7 = f.add_subplot(5,2,7,axisbg='none')
    plot_posterior( posterior_std2, bins=bin_edges_stds, ax=ax7,
                    title='Group 2 Std. Dev.',
                    label=r'$\sigma_2$')

    ax9 = f.add_subplot(5,2,9,axisbg='none')
    plot_posterior( lognup, bins=n_bins, ax=ax9,
                    title='Normality',
                    label=r'$\mathrm{log10}(\nu)$')

    ax6 = f.add_subplot(5,2,6,axisbg='none')
    plot_posterior( diff_means, bins=n_bins, ax=ax6,
                    title='Difference of Means',
                    stat='mean',
                    draw_zero=True,
                    label=r'$\mu_1 - \mu_2$')

    ax8 = f.add_subplot(5,2,8,axisbg='none')
    plot_posterior( diff_stds, bins=n_bins, ax=ax8,
                    title='Difference of Std. Dev.s',
                    draw_zero=True,
                    label=r'$\sigma_1 - \sigma_2$')

    ax10 = f.add_subplot(5,2,10,axisbg='none')
    plot_posterior( effect_size, bins=n_bins, ax=ax10,
                    title='Effect Size',
                    draw_zero=True,
                    label=r'$(\mu_1 - \mu_2)/\sqrt{(\sigma_1 - \sigma_2)/2}$')

    orig_vals = np.concatenate( (M.group1.value, M.group2.value) )
    bin_edges = np.linspace( np.min(orig_vals), np.max(orig_vals), 30 )

    ax2 = f.add_subplot(5,2,2,axisbg='none')
    plot_data_and_prediction(M.group1.value, posterior_mean1, posterior_std1,
                             post_nu_minus_one, ax=ax2, bins=bin_edges, group=1)

    ax4 = f.add_subplot(5,2,4,axisbg='none',sharex=ax2,sharey=ax2)
    plot_data_and_prediction(M.group2.value, posterior_mean2, posterior_std2,
                             post_nu_minus_one, ax=ax4, bins=bin_edges, group=2)


    f.subplots_adjust(hspace=0.82,top=0.97,bottom=0.06,
                      left=0.09, right=0.95, wspace=0.45)
    return f

