data {
    int<lower=0> num_obs;                                       // number of observations
    vector[num_obs] event_counts;                               // number of events
    array[num_obs] int<lower=0,upper=1> fatality_flag;         // fatality indicator
}

parameters{
    real alpha;
    real beta_;
}

model{
    // priors
    alpha ~ normal(0, 1);
    beta_ ~ normal(0, 1);

    // likelihood
    fatality_flag ~ bernoulli_logit(alpha + beta_ * event_counts);
}