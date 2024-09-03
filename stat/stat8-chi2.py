import numpy as np
mat_a = np.array([14, 4, 0, 10]).reshape(2, 2)

from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(2)
p.round(4)

1 - chi2.cdf()