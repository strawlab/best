"""
Comparison of two groups with paired samples with BEST.

This is the Bayesian equivalent of a paired difference test:
we have data on every individual from two arms of an experiment,
or from two points in time, and we want to assess the magnitude
of the increase from the morning to the evening.
"""
import numpy as np
import best


morning = [8.99, 9.21, 9.03, 9.15, 8.68, 8.82, 8.66, 8.82, 8.59, 8.14,
           9.09, 8.80, 8.18, 9.23, 8.55, 9.03, 9.36, 9.06, 9.57, 8.38]
evening = [9.82, 9.34, 9.73, 9.93, 9.33, 9.41, 9.48, 9.14, 8.62, 8.60,
           9.60, 9.41, 8.43, 9.77, 8.96, 9.81, 9.75, 9.50, 9.90, 9.13]

print('Performing Bayesian analysis of the two groups.\n'
      'This might take a while...')

best_out = best.analyze_one(np.subtract(evening, morning))

fig = best.plot_all(best_out, group_name='Evening–morning diff.:')
fig.savefig('paired_samples.png', dpi=120)
fig.savefig('paired_samples.pdf')
