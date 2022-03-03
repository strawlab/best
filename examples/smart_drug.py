"""
Comparison of two groups with independent samples with BEST.

This example reproduces Figure 3 from (Kruschke, 2012). Output is saved as
 "smart_drug.png" and "smart_drug.pdf"

Kruschke, J. K. (2012) Bayesian Estimation Supersedes the t Test.
        Journal of Experimental Psychology: General.
        v.142(2), pp.573-603. (doi: 10.1037/a0029146)
"""

import best


# IQ scores of those who took the smart drug
study = [101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103,
         100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101, 102, 102,
         103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124, 101, 101, 100,
         101, 101, 104, 100, 101]

# IQ scores of those who took a placebo pill
control = [99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105,
           88, 101, 100, 104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100,
           101, 99, 101, 100, 100, 101, 100, 99, 101, 100, 102, 99, 100, 99]

print('Performing Bayesian analysis of the two groups.\n'
      'This might take a while...')

best_out = best.analyze_two(study, control)

fig = best.plot_all(best_out, group1_name='Study group', group2_name='Control group')
fig.savefig('smart_drug.png', dpi=120)
fig.savefig('smart_drug.pdf')
