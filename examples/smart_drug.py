import best
import best.plot
from pymc import MCMC

# This example reproduces Figure 3 of
#
# Kruschke, J. (2012) Bayesian estimation supersedes the t
#     test. Journal of Experimental Psychology: General.
#
# According to the article, the data were generated from t
# distributions of known values.

drug = (101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 100, 95, 102, 106,
        109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97, 103, 101, 97, 104,
        96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101)
placebo = (99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105, 88, 101, 100,
           104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100, 101, 99, 101, 100, 100,
           101, 100, 99, 101, 100, 102, 99, 100, 99)

data = {'drug': drug, 'placebo': placebo}

model = best.make_model(data)

M = MCMC(model)
M.sample(iter=110000, burn=10000)

fig = best.plot.make_figure(M)
fig.savefig('smart_drug.png', dpi=70)
