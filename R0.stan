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
}

parameters {
  /* Serial time (days) */
  real<lower=0> tau;

  /* Gaussian process noise */
  real<lower=0> sigma;

  /* Expected counts each day */
  real<lower=0> exp_cts[ndays];
}

transformed parameters {
  /* Reproduction rate */
  real Rt[ndays-1];

  for (i in 2:ndays) {
    Rt[i-1] = tau*(log(exp_cts[i]) - log(exp_cts[i-1])) + 1;
  }
}

model {
  /* Prior on serial time is log-normal with mean and s.d. matching input */
  tau ~ lognormal(tau_mu, tau_sigma);

  /* Prior on sigma; scatter by a factor of 3 around 0.3 per day */
  sigma ~ lognormal(log(0.3), 1);

  /* Prior on first day's expected counts is broad log-normal */
  exp_cts[1] ~ lognormal(log(10), 2);

  /* The AR(1) process prior; we begin with an N(0,1) prior on Rt at the first
     sample, and then increment according to the AR(1) process.  We need to
     include a Jacobian factor because we sample in exp_cts, not Rt:

     Jac = d(Rt[i]) / d(exp_cts[i+1]) = tau / exp_cts[i+1]

     We accumulate the -log(exp_cts[i+1]) as we calculate the Rt priors; then we
     account for the (ndays-1)*log(tau) outside the loop.  */
  Rt[1] ~ normal(0, 10);
  target += -log(exp_cts[2]);
  for (i in 2:ndays-1) {
    Rt[i] ~ normal(Rt[i-1], sigma);
    target += -log(exp_cts[i+1]);
  }
  target += (ndays-1)*log(tau);

  /* Poisson likelihood for the counts on each day. */
  k ~ poisson(exp_cts);
}
