data {
  int<lower=0> num_points;             // Number of observations
  vector[num_points] event_counts;     // Covariate
  array[num_points] int<lower=0,upper=1> fatalities_flag; // Binary response
  array[num_points] int<lower=0,upper=1> prev_fatalities_flag; // Binary response
}

parameters {
  real alpha;   // Intercept
  real beta_;    // Coefficient for event_counts
  real gamma_;   // Coefficient for prev_fatalities_flag

//   real p3
}

model {
  // Priors (can be modified according to your beliefs)
  alpha ~ normal(0, 1);
  beta_ ~ normal(0, 1);
  gamma_ ~ normal(0, 1);

  // Logistic regression model
  for (n in 1:num_points) {
    // if event_counts[n] == 2:
    // fatalities_flag ~ bin(1,p3*q2)
    fatalities_flag[n] ~ bernoulli_logit(alpha + beta_ * event_counts[n] + gamma_*prev_fatalities_flag[n]);
  }
  
  // fatalities ~ bernoulli_logit(alpha + beta_ * event_counts + gamma_*prev_fatalities_flag); // Fully vectorised version
}
