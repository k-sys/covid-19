# Original R_t code https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
from datetime import datetime as dt
from datetime import timedelta
import logging

import pandas as pd
import numpy as np
import requests

from scipy import stats as sps

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s %(name)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

SKIP_N_LAST_DAYS_IN_DATA = 5

R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
GAMMA = 1 / 4

state_name = 'Finland'  # TODO, it is possible to calculate all Finn states separately


def get_data_from_THL():
    url = "https://w3qa5ydb4l.execute-api.eu-west-1.amazonaws.com/prod/processedThlData"

    payload = ""
    response = requests.request("GET", url, data=payload)

    logger.info(f'Response status: {response.status_code}')
    response_json = response.json()

    finland = pd.DataFrame(response_json['confirmed']['Kaikki sairaanhoitopiirit'])
    finland['date'] = finland['date'].apply(pd.to_datetime).dt.date
    finland.set_index(finland['date'], inplace=True)

    last_day = dt.today().date() - timedelta(SKIP_N_LAST_DAYS_IN_DATA)
    logger.info(f'Use data before {last_day}')
    return finland[finland['date'] < last_day]


def prepare_cases(cases):
    new_cases = cases

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


if __name__ == '__main__':
    logger.info('START')

    finland = get_data_from_THL()
    cases = finland['value'].rename(f"{state_name} cases")

    # TODO write original_cases, smoothed_case to files in blob storage
    original_cases, smoothed_cases = prepare_cases(cases)

    posteriors = get_posteriors(smoothed_cases)
    hdis = highest_density_interval(posteriors)

    most_likely = posteriors.idxmax().rename('ML')

    result = pd.concat([most_likely, hdis], axis=1).reset_index()

    # TODO write result to file in blob storage
    filename = f'Rt_{dt.today().date()}.tsv'
    logger.info(f'TODO: write file to {filename} in blob storage')
    print(result)
    logger.info('DONE')
