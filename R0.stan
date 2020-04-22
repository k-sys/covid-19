data {
  /* new infection counts over days */
  int ndays;
  int k[ndays];

  /* Serial time */
  real lambda;
}

parameters {
  /* Gaussian process noise */
  real<lower=0> sigma;

  /* Expected number of counts at day zero. */
  real<lower=0> c0;

  /* Per-person per infectious interval number of transmissions; applies to day
  /* 2...N */
  real<lower=0> Rt[ndays-1];
}

transformed parameters {
  /* Expected number of positive samples on each day. */
  real exp_cts[ndays];

  exp_cts[1] = c0;
  for (i in 2:ndays) {
    exp_cts[i] = exp_cts[i-1]*exp(lambda*(Rt[i-1]-1));
  }
}

model {
  /* Prior on sigma; scatter by a factor of 3 around 0.3 per day */
  sigma ~ lognormal(log(0.3), 1);

  /* First day is w/i a factor of three of having 10 positive counts. */
  c0 ~ lognormal(log(10), 1);

  /* The AR(1) process prior; we begin with an N(0,1) prior on Rt at the first
  /* sample, and then increment according to the AR(1) process */
  Rt[1] ~ normal(0, 10);
  for (i in 2:ndays-1) {
    Rt[i] ~ normal(Rt[i-1], sigma);
  }

  /* Poisson likelihood for the counts on each day. */
  k ~ poisson(exp_cts);
}
