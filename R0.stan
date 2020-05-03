data {
  /* new infection counts over days */
  int ndays;
  int k[ndays];

  /* Parameters for the marginalization over serial time */
  real tau_mean;
  real tau_std;

  /* sigma is given an N(0, scale) prior. */
  real sigma_scale;
}

transformed data {
  /* Parameters for a log-normal distribution on the serial time that match the
  /* given mean and s.d. */
  real tau_mu = log(tau_mean / sqrt(1.0 + tau_std^2/tau_mean^2));
  real tau_sigma = sqrt(log1p(tau_std^2/tau_mean^2));
}

parameters {
  /* Serial time (days) */
  real<lower=0> tau;

  /* stepsize of the random walk in log(Rt) */
  real<lower=0> sigma;

  /* First day's number of cases */
  real<lower=0> L0;

  /* Will be transformed into Rt. */
  real Rt_raw[ndays-1];
}

transformed parameters {
  real Rt[ndays-1];
  real log_jacobian;

  {
    real log_jac[ndays-1];
    real exp_cts[ndays];

    exp_cts[1] = k[1];
    for (i in 2:ndays) {
      real Rt_counts_raw;
      real Rt_counts;
      real sd_counts;
      real wt_counts;

      real Rt_prior;
      real sd_prior;
      real wt_prior;

      real Rt_total;
      real sd_total;
      real wt_total;

      real Rt_est;

      if (i == 2) {
        Rt_prior = 3.0;
        sd_prior = 2.0;
      } else {
        Rt_prior = Rt[i-2];
        sd_prior = sigma*Rt[i-2];
      }
      wt_prior = 1.0/(sd_prior*sd_prior);

      Rt_counts_raw = tau*(log(k[i]) - log(exp_cts[i-1])) + 1.0;
      Rt_counts = log_sum_exp(Rt_counts_raw, 0.0); /* Ensure Rt_counts > 0, and make it linear for Rt_counts_raw > 0.1 or so */
      if (Rt_counts_raw > 0) {
        sd_counts = tau/sqrt(k[i]+1)/(1.0 + exp(-Rt_counts_raw));
      } else {
        sd_counts = tau/sqrt(k[i]+1)*exp(Rt_counts_raw)/(1.0 + exp(Rt_counts_raw));
      }
      wt_counts = 1.0/(sd_counts*sd_counts);

      wt_total = wt_counts + wt_prior;
      Rt_total = (wt_counts*Rt_counts + wt_prior*Rt_prior)/wt_total;
      sd_total = sd_counts*sd_prior/sqrt(sd_counts*sd_counts + sd_prior*sd_prior);

      Rt[i-1] = Rt_total*exp(sd_total/Rt_total*Rt_raw[i-1]);
      log_jac[i-1] = log(Rt[i-1]) + log(sd_total) - log(Rt_total);
      exp_cts[i] = exp_cts[i-1]*exp((Rt[i-1]-1)/tau);
    }

    log_jacobian = sum(log_jac);
  }
}

model {
  real exp_cts[ndays];

  /* Prior on serial time is log-normal with mean and s.d. matching input */
  tau ~ lognormal(tau_mu, tau_sigma);

  /* Prior on sigma, supplied by the user. */
  sigma ~ normal(0, sigma_scale);

  /* The AR(1) process prior; we begin with an N(3,2) prior on Rt based on
     Chinese studies at the first sample, and then increment according to the
     AR(1) process.  Above, we have computed the Jacobian factor between Rt and
     rate_raw (which we sample in). */
  Rt[1] ~ lognormal(log(3), 2.0/3.0);
  for (i in 2:ndays-1) {
    Rt[i] ~ lognormal(log(Rt[i-1]), sigma);
  }
  target += log_jacobian;

  exp_cts[1] = L0;
  for (i in 2:ndays) {
    exp_cts[i] = exp_cts[i-1]*exp((Rt[i-1]-1)/tau);
  }
  /* Poisson likelihood for the counts on each day. */
  k ~ poisson(exp_cts);
}
