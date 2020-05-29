data {
  int ndays;
  int k[ndays]; /* Positive tests */
  int n[ndays]; /* Number of tests */

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

  /* The parameter phi controls the "excess variance" of the negative binomial
  /* over Poisson.  For the negative binomial distribution, we have

  <X> = mu

  Var(x) = mu + mu^2/phi = mu*(1 + mu/phi)

  So that for mu << phi the distribution behaves "like Poisson;" but for mu >>
  phi we have

  sqrt(Var(x))/<X> = 1/sqrt(phi)

  that is: the *fractional* uncertainty asymptotes to the same fractional
  uncertainty as Possion with phi counts, no matter how large mu grows.

  Here we choose phi = 1000, semi-arbitrarily.  This limits the relative
  uncertainty in the positive rate to ~few percent on any given day.

  */
  real phi = 100.0;
}

parameters {
  /* Serial time (days) */
  real<lower=0> tau;

  /* stepsize of the random walk in log(Rt) */
  real<lower=0> sigma;

  /* First day's expected number of infections. */
  real L0_raw;

  /* Will be transformed into Rt. */
  real Rt_raw[ndays-1];
}

transformed parameters {
  real L0;
  real log_jac_L0;
  real Rt[ndays-1];
  real log_jacobian;

  L0 = (k[1]+1)*exp(L0_raw/sqrt(k[1]+1));
  log_jac_L0 = log(L0) - 0.5*log(k[1]+1);

  /* Here we transform the raw variables into Rt following the AR(1) prior;
     because we are using a negative binomial observational model, as long as
     the value of sigma is comparable to 1/sqrt(phi) ~ 0.03 (i.e. as long as the
     user puts in a small sigma_scale), then the prior is comparable to the
     likelihood for each observation, and we will be approximately uncorrelated. */
  {
    real log_jac[ndays-1];

    for (i in 2:ndays) {
      if (i == 2) {
        Rt[i-1] = 3*exp(2.0/3.0*Rt_raw[i-1]);
        log_jac[i-1] = log(Rt[i-1]) + log(2.0/3.0);
      } else {
        Rt[i-1] = Rt[i-2]*exp(sigma*Rt_raw[i-1]);
        log_jac[i-1] = log(Rt[i-1]) + log(sigma);
      }
    }

    log_jacobian = sum(log_jac);
  }
}

model {
  real ex_cts[ndays];

  /* Prior on serial time is log-normal with mean and s.d. matching input */
  tau ~ lognormal(tau_mu, tau_sigma);

  /* Prior on sigma, supplied by the user. */
  sigma ~ normal(0, sigma_scale);

  /* Initial log-odds given wide prior */
  L0 ~ lognormal(log(10), 1);
  target += log_jac_L0;

  /* The AR(1) process prior; we begin with an N(3,2) prior on Rt based on
     Chinese studies at the first sample, and then increment according to the
     AR(1) process.  Above, we have computed the Jacobian factor between Rt and
     rate_raw (which we sample in). */
  Rt[1] ~ lognormal(log(3), 2.0/3.0);
  for (i in 2:ndays-1) {
    Rt[i] ~ lognormal(log(Rt[i-1]), sigma);
  }
  target += log_jacobian;

  ex_cts[1] = L0;
  for (i in 2:ndays) {
    ex_cts[i] = ex_cts[i-1]*exp((Rt[i-1]-1.0)/tau);
  }

  /* Negative binomial likelihood for the counts on each day. */
  k ~ neg_binomial_2(ex_cts, phi);
}
