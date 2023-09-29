functions {
   real product_from_index(array[] real x, int initial_index){
    real result = 1.0;
    for (i in initial_index:size(x)){
        result *= x[i];
    }
    return result;
   }
}

data {
    int<lower=0> num_obs;                                       // number of observations
    array[num_obs] int<lower=0> event_bins;           // number of events
    array[num_obs] int<lower=0,upper=1> fatality_flag;         // fatality indicator
    int<lower=0> max_events;                                  // maximum weekly events, also referred to as K
}

parameters{
    real<lower=0,upper=1>pK;
    array[max_events - 1] real<lower=0,upper=1> q;
}

model{
    // priors
    pK ~ beta(1,1);

    for (k in 1:max_events-1){
        q[k] ~ beta(1,1);
    }

    // likelihood
    for (n in 1:num_obs){
        if (event_bins[n] == max_events) {
            fatality_flag[n] ~ bernoulli(pK);
        } else if (event_bins[n] > 0) {
            real qi = product_from_index(q, event_bins[n]);
            fatality_flag[n] ~ bernoulli(pK * qi);
        }
    }
}

generated quantities {
   array[max_events-1] real<lower=0,upper=1> p;

   for (i in 1:max_events-1){
    p[i] = pK * product_from_index(q, i);
   }

}