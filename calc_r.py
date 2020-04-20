# Original code https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
GAMMA = 1/4

state_name = 'Finland' # TODO, it is possible to calculate all Finnis state separetely


def prepare_cases(cases):
    new_cases = cases  # cases.diff() # HUOM

    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def get_posteriors(sr, window=7, min_periods=1):
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.

    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))

    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data=np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index=r_t_range,
        columns=sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)

    return posteriors


def highest_density_interval(pmf, p=.95):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i + 1:]):
            if (high_value - value > p) and (not best or j < best[1] - best[0]):
                best = (i, i + j + 1)
                break

    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=['Low', 'High'])


# READ DATA
# TODO: get new data from APIs
finland = pd.read_csv('finland_stat.csv', sep='\t',
                      skiprows=1, header=None, names=['date', 'cases'],
                      parse_dates=['date'])

# set index
finland.set_index(finland['date'], inplace=True)

# cases = states.xs(state_name).rename(f"{state_name} cases")
cases = finland['cases'].rename(f"{state_name} cases")

original_cases, smoothed_cases = prepare_cases(cases)

# TODO write cases to file

posteriors = get_posteriors(smoothed_cases)
hdis = highest_density_interval(posteriors)

most_likely = posteriors.idxmax().rename('ML')

# Look into why you shift -1
result = pd.concat([most_likely, hdis], axis=1)

# TODO write result (ML, Low, High) to file
print(result)
