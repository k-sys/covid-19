# Original R_t code https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
from datetime import datetime as dt
from datetime import timedelta
import logging
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import pandas as pd
import numpy as np
import requests

from scipy import stats as sps

import os

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s %(name)s:%(lineno)d - %(message)s")
logger = logging.getLogger(__name__)

SKIP_N_LAST_DAYS_IN_DATA = 5

R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
GAMMA = 1 / 4

state_name = 'Finland'  # TODO, it is possible to calculate all Finn states separately

# Setup Blob Service client
CONTAINER_NAME = "estimate-rt"
CONNECTION_STRING = "" # TODO!!! NOTE CONNECTION_STRING!!

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


def plot_rt(result, ax, state_name, fig):
    ax.set_title(f"{state_name}")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(np.r_[
                              np.linspace(BELOW, MIDDLE, 25),
                              np.linspace(MIDDLE, ABOVE, 25)
                          ])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5

    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values

    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')

    highfn = interp1d(date2num(index),
                      result['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1] + pd.Timedelta(days=1))

    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 3.5)
    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')


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


def main():
    logger.info('START')
    fname_date = str(dt.today()).replace(' ', '_')

    finland = get_data_from_THL()
    cases = finland['value'].rename(f"{state_name} cases")

    original_cases, smoothed_cases = prepare_cases(cases)

    # # Create Blob service connection
    # blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    # container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    cases_filename = f'{fname_date}_cases.csv'

    os.mkdir(fname_date)
    original_cases.plot(title=f"{state_name} New Cases per Day",
                  c='k',
                  linestyle=':',
                  alpha=.5,
                  label='Actual',
                  legend=True,
                  figsize=(600 / 72, 400 / 72))

    ax = smoothed_cases.plot(label='Smoothed',
                       legend=True)
    ax.get_figure().set_facecolor('w')
    plt.savefig(f'{fname_date}/{fname_date}_cases.png')

    # logger.info(f'Upload to Azure cases_filename')

    # blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=cases_filename)
    # blob_client.upload_blob(pd.concat([original_cases.rename('Original Cases'),
    #                                    smoothed_cases.rename('Smoothed Cases')], axis=1).to_csv(index_label='date'))

    logger.info('Calculate R_t')
    posteriors = get_posteriors(smoothed_cases)
    hdis = highest_density_interval(posteriors)

    most_likely = posteriors.idxmax().rename('ML')

    result = pd.concat([most_likely, hdis], axis=1) #.reset_index()

    result_filename = f'{fname_date}_Rt.csv'
    # logger.info(f'Write file to {result_filename} in blob storage')

    # blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=result_filename)
    # blob_client.upload_blob(result.to_csv(index=False))
    fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))

    plot_rt(result, ax, state_name, fig)
    ax.set_title(f'Real-time $R_t$ for {state_name}')
    ax.set_ylim(.5, 3.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.savefig(f'{fname_date}/{fname_date}_Rt.png')

    logger.info('DONE')


main()