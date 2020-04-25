data {
  /* new infection counts over days */
  int ndays;
  int k[ndays];

  /* Parameters for the marginalization over serial time */
  real tau_mean;
  real tau_std;
}

transformed data {
  /* Parameters for a log-normal distribution on the serial time that match the
  /* given mean and s.d. */
  real tau_mu = log(tau_mean / sqrt(1.0 + tau_std^2/tau_mean^2));
  real tau_sigma = sqrt(log1p(tau_std^2/tau_mean^2));

  real Rt_est = 3.0;
  real log_first_factor = (Rt_est-1)/tau_mean;
}

parameters {
  /* Serial time (days) */
  real<lower=0> tau;

  real<lower=0> smoothing_timescale;

  /* Gaussian process noise */
  real log_sigma_raw;

  /* Transformed to expected counts each day */
  real rate_raw[ndays];
}

transformed parameters {
  real sigma = tau * exp(log_sigma_raw - log(100));
  real exp_cts[ndays];
  real Rt[ndays-1];
  real log_jacobian;

  {
    real sf = exp(-1.0/smoothing_timescale);
    real log_jacs[ndays];

    exp_cts[1] = k[1]*exp(rate_raw[1]/sqrt(k[1]));
    log_jacs[1] = log(exp_cts[1]) - 0.5*log(k[1]);

    for (i in 2:ndays) {
      if (k[i] > 100) {
        /* Then we transform rate_raw to exp_cts, because the cts are pretty well constrained. */
        exp_cts[i] = k[i]*exp(rate_raw[i]/sqrt(k[i]));

        Rt[i-1] = tau*(log(exp_cts[i]) - log(exp_cts[i-1])) + 1;
        log_jacs[i] = log(tau) - 0.5*log(k[i]);
      } else {
        /* The counts are not well constrained, so we produce Rt instead. */
        if (i == 2) {
          Rt[i-1] = tau*(log_first_factor + rate_raw[i]/tau_mean) + 1;
          log_jacs[i] = log(tau) - log(tau_mean);
        } else {
          Rt[i-1] = sf*Rt[i-2] + sigma*rate_raw[i];
          log_jacs[i] = log(sigma);
        }
        exp_cts[i] = exp_cts[i-1]*exp((Rt[i-1]-1)/tau);
      }
    }

    log_jacobian = sum(log_jacs);
  }
}

model {
  real sf = exp(-1.0/smoothing_timescale);

  /* Prior on serial time is log-normal with mean and s.d. matching input */
  tau ~ lognormal(tau_mu, tau_sigma);

  /* Prior on sigma; scatter by a factor of 3 around 0.3 per day; Jacobian is

  d(sigma) / d(log_sigma_raw) = sigma
   */
  sigma ~ lognormal(log(0.3), 1);
  target += log(sigma);

  /* Smooth on timescale ~14 days but with a lot of uncertainty. */
  smoothing_timescale ~ lognormal(log(14), 0.5);

  /* Prior on first day's expected counts is broad log-normal */
  exp_cts[1] ~ lognormal(log(k[1]), 1);

  /* The AR(1) process prior; we begin with an N(3,2) prior on Rt based on
     Chinese studies at the first sample, and then increment according to the
     AR(1) process.  Above, we have computed the Jacobian factor between Rt and
     rate_raw (which we sample in). */
  Rt[1] ~ normal(3, 2);
  for (i in 2:ndays-1) {
    Rt[i] ~ normal(sf*Rt[i-1], sigma);
  }

  target += log_jacobian;

  /* Poisson likelihood for the counts on each day. */
  k ~ poisson(exp_cts);
}
