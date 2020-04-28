data {
  int ndays;
  int k[ndays]; /* Number of positive tests. */
  int n[ndays]; /* Total number of tests. */

  /* mu and sigma parameters of the log-normal dist applied to serial time (mean
  /* and s.d. of log(serial time)) */
  real tau_mu;
  real tau_sigma;

  /* same, but for the parameter that is the s.d. of the increments to log-Rt
  /* day-to-day */
  real sigma_mu;
  real sigma_sigma;
}

transformed data {
  real log_odds_est[ndays];

  for (i in 1:ndays) {
    log_odds_est[i] = log(k[i]+1.0) - log(n[i]-k[i]+1.0);
  }
}

parameters {
  /* Serial time */
  real<lower=0> tau;

  /* S.d. of the scatter in log(R_t) day-to-day */
  real<lower=0> sigma;

  /* Starting p for the Binomial counts on the first day. */
  real logit_p0;

  /* Will be transformed into R_t below; it is more efficient to sample in
     Rt_raw.  (That is, we arrange the map from Rt_raw to Rt so that Rt_raw has
     a distribution closer to iid N(0,1)). */
  real Rt_raw[ndays-1];
}

transformed parameters {
  /* Transformed from Rt_raw to make sampling more efficient. */
  real Rt[ndays-1];

  /* Accumulated log-Jacobian from the transformation; needed so that we can
  /* impose a prior on Rt below even though we sample in Rt_raw. */
  real log_jacobian;

  /* Records the log_odds timeseries for later comparison to data. */
  real log_odds[ndays];

  {
    real log_jac[ndays-1];

    /* The transformation depends on whether we expect to have a good (that is,
       better than the prior) constraint on Rt from the number of observations;
       if we do, then we will be likelihood dominated, and we use Rt_raw to
       scatter around the estimated Rt; if we do not, then we expect to be
       prior-dominated, and we use Rt_raw as the *increment* in log(Rt) so that
       it is iid N(0,1) by the prior.

       The choice of 100 samples as the cutoff between likelihood-dominated and
       prior-dominated is based on preliminary runs that suggest a typical scale
       for sigma ~ 0.1, so a 10% day-to-day fluctuation in Rt will be the usual
       prior bound; with 100 samples we therefore start to become
       likelihood-dominated. */
    log_odds[1] = logit_p0;

    Rt[1] = 3*exp(2.0/3.0*Rt_raw[1]);
    log_jac[1] = log(Rt[1]) + log(2.0/3.0);
    log_odds[2] = log_odds[1] + (Rt[1]-1.0)/tau;
    for (i in 2:ndays-1) {
      if (k[i] > 200) {
        /* Likelihood dominated */
        real log_odds_target = log_odds_est[i+1] + Rt_raw[i]/sqrt(k[i]);
        real Rt_est = tau*(log_odds_target - log_odds[i]) + 1.0;
        /* This trick ensures that Rt > 0; it is a smooth form of "max", which
        /* is the identity for Rt > 0.5. */
        Rt[i] = log_sum_exp(2.0*Rt_est, 0)/2.0;
        log_jac[i] = 2.0*(Rt_est - Rt[i]) + log(tau) - 0.5*log(k[i]);
        log_odds[i+1] = log_odds[i] + (Rt[i]-1)/tau;
      } else {
        Rt[i] = Rt[i-1]*exp(sigma*Rt_raw[i]);
        log_jac[i] = log(Rt[i]) + log(sigma);
        log_odds[i+1] = log_odds[i] + (Rt[i]-1)/tau;
      }
    }

    log_jacobian = sum(log_jac);
  }
}

model {
  tau ~ lognormal(tau_mu, tau_sigma);
  sigma ~ lognormal(sigma_mu, sigma_sigma);
  logit_p0 ~ normal(log((k[1]+1.0) / (n[1]-k[1]+1.0)), 1);

  Rt[1] ~ lognormal(log(3), 2.0/3.0);
  for (i in 2:ndays-1) {
    Rt[i] ~ lognormal(log(Rt[i-1]), sigma);
  }
  target += log_jacobian;

  for (i in 1:ndays) {
    if (n[i] > 0) {
      k[i] ~ binomial_logit(n[i], log_odds[i]);
    }
  }
}
