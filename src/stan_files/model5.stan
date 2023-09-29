data {
    int<lower=0> num_obs;                                       // number of observations
    int<lower=0> num_covs;                                      // maximum number of covariates
    matrix[num_obs, num_covs] observations;                     // observations matrix
    array[num_covs] int<lower=0,upper=1> inclusion;             // covariate inclusion array
    array[num_obs] int<lower=0, upper=1> responses;             // binary response variable
}

parameters {
   real alpha;
   array[num_covs] real beta_;
}

model {
   vector[num_obs] log_odds;

   alpha ~ normal(0,1);
   beta_ ~ normal(0,1);

   for (obs in 1:num_obs) {
    log_odds[obs] = alpha;
    for (cov in 1:num_covs) {
        if (inclusion[cov] == 1) {
            log_odds[obs] += beta_[cov] * observations[obs, cov];
        }
    }
   }

   responses ~ bernoulli_logit(log_odds);
}