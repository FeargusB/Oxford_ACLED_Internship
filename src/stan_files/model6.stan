data {
    int<lower=0> num_obs;                          
    vector[num_obs] event_counts;                               
    array[num_obs] int<lower=0,upper=1> fatality_flag;
}

parameters{
    real<lower=0,upper=1> p;
}

model{
    // priors
    p ~ beta(1, 1);

    for (n in 1:num_obs){
        fatality_flag[n] ~ bernoulli(1 - (1-p)^event_counts[n]);
    }
}