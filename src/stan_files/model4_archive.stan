data {
    int<lower=0> num_obs;                                       // number of observations
    vector[num_obs] event_bins;                               // number of events
    array[num_obs] int<lower=0,upper=1> fatality_flag;         // fatality indicator
}

parameters{
    real<lower=0,upper=1>p3;
    real<lower=0,upper=1>q2;
    real<lower=0,upper=1>q1;
}

model{
    // priors
    p3 ~ beta(1,1);
    q2 ~ beta(1,1);
    q1 ~ beta(1,1);

    // likelihood
    for (n in 1:num_obs){
        if (event_bins[n] == 3) {
            fatality_flag[n] ~ bernoulli(p3);
        } else if (event_bins[n] == 2) {
            fatality_flag[n] ~ bernoulli(p3 * q2);
        } else if (event_bins[n] == 1) {
            fatality_flag[n] ~ bernoulli(p3 * q2 * q1);
        }
    }
}

generated quantities {
   real<lower=0,upper=1> p2;
   real<lower=0,upper=1> p1;

   p2 = p3*q2;
   p1 = p3*q2*q1;
}