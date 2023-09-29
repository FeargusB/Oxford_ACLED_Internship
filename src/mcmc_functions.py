import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

def logit_mh_mcmc(responses, observed_covariates,
                    parameter_names, initial_parameters, parameter_stepsizes, hyperparameters, proposal_dists,
                    target_dist, iterations=10000):
    '''
    A Metropolis-Hastings algorithm for a logistic model with a free number of parameters
    Currently assumes a symmetric proposal distribution for simplicity

    Arguments:
    responses - a vector of N observed responses
    observed_covariates - an array of N observed covariate values (rows) for M covariates (columns)
    parameter_names - a list of K string parameter names
    initial_parameters - a list of K float initial parameter values
    parameter_stepsizes - a list of K float initial parameter stepsizes
    parameter_prior_parameters - a list of K float parameters for the prior distributions
    proposal_dists - a list of K functions, representing proposal distributions, taking the current parameter value and a stepsize as arguments
    target_dist - a function to be approximated by the MH MCMC algorithm
    iterations - an integer number of iterations to perform

    Returns:
    a pandas data frame with K columns of iterations+1 sampled parameter values from the target distribution
    '''

    ## INITIAL SETUP

    # find the number of parameters
    num_params = len(parameter_names)

    # create a matrix to store the sampled values
    params_chain = np.zeros((iterations+1, num_params))
    params_chain[0,:] = initial_parameters

    # initialise parameters and acceptance rates
    current_params = initial_parameters
    param_acceptance_count = np.zeros(num_params)        

    ## MAIN CHAIN SECTION

    # perform the required number of iterations
    for i in range(iterations):
        # update the parameters sequentially
        for idx, param in enumerate(current_params):
            # propose the same parameters, changing the appropriate parameter only
            # this is a messy method but I'd not realised that lists aren't immutable
            proposed_param = proposal_dists[idx](param, parameter_stepsizes[idx])
            proposed_params = np.zeros(num_params)
            proposed_params[:idx] = current_params[:idx]
            proposed_params[idx] = proposed_param
            if idx != num_params-1:
                proposed_params[idx+1:] = current_params[idx+1:]

            # find the target log densities under the current and proposed parameters, and the acceptance ratio
            log_post_current = target_dist(current_params, hyperparameters, responses, observed_covariates)
            log_post_proposed = target_dist(proposed_params, hyperparameters, responses, observed_covariates)
            acceptance_ratio = log_post_proposed - log_post_current

            if acceptance_ratio > np.log(np.random.rand()):
                # if accepted, record the new parameter and update the parameters
                params_chain[i+1, idx] = proposed_param
                current_params = proposed_params
                param_acceptance_count[idx] += 1

            else:
                params_chain[i+1, idx] = param
    
    for idx, param in enumerate(parameter_names):
        print(f'Proposal acceptance rate of {param_acceptance_count[idx]/iterations} for parameter {param}')

    params_chain_frame = pd.DataFrame(params_chain, columns = parameter_names)

    return params_chain_frame

def logit_lpl(parameters, hyperparameters):
    '''
    Log prior likelihood function for a flexible number of parameters
    Each parameter has a Normal prior distribution mean 0, specified st.dev.

    Arguments:
    parameters - a list of parameters whose log prior likelihood is to be found
    hyperparameters - a list of standard deviations to use in the Normal prior for each parameter

    Returns the log prior likelihood for the parameters under the specified Normal distribution
    '''
    lhood = 0
    for idx, param in enumerate(parameters):
        lhood += norm.logpdf(param, loc=hyperparameters[idx][0], scale=hyperparameters[idx][1])
    return lhood

def general_logit(covariates, parameters):
    '''
    Multi-covariate logistic function

    Arguments:
    covariates - a list of n-1 covariate values
    parameters - a list of n parameter values

    Returns the evaluation of the logistic function with parameters params at the specified covariates
    '''
    odds = parameters[0]
    for idx, cov in enumerate(covariates):
        odds += cov*parameters[idx+1]
    return np.exp(odds)/(1+np.exp(odds))

def single_logit(covariate, parameters):
    '''
    Single covariate logistic function

    Arguments:
    covariate - an integer-like object
    parameters - a list of 2 parameters

    Returns the evaluation of the logistic function with the specified parameters at the covariate
    '''
    odds = parameters[0] + covariate*parameters[1]
    return np.exp(odds)/(1+np.exp(odds))

# def logit_ll(parameters, responses, observed_covariates):
#     '''
#     Log-likelihood function for the logistic model

#     Arguments:
#     responses - a list or numpy array length N of binary responses
#     observed_covariates - a numpy matrix with N rows of covariate observations and M columns of covariate types
#     parameters - a list of M+1 parameter values

#     Returns the log-likelihood of the responses given the covariates and parameter values
#     '''
#     lhoods = np.zeros(len(responses))
#     if observed_covariates.ndim == 1:
#         for idx, resp in enumerate(responses):
#             cov = observed_covariates[idx]
#             prob = single_logit(cov, parameters)
#             lhoods[idx] = resp*np.log(prob) + (1-resp)*np.log(1-prob)
#     else:
#         for idx, resp in enumerate(responses):
#             covs = observed_covariates[idx,:]
#             prob = general_logit(covs, parameters)
#             lhoods[idx] = resp*np.log(prob) + (1-resp)*np.log(1-prob)
#     return np.sum(lhoods)

def logit_ll(parameters, responses, covariates):
    '''
    Log-likelihood function for the logistic model

    Arguments:
    responses - a list or numpy array length N of binary responses
    covariates - a numpy matrix with N rows of covariate observations and M columns of covariate types
    parameters - a list of M+1 parameter values

    Returns the log-likelihood of the responses given the covariates and parameter values
    '''
    lhoods = np.zeros(len(responses))
    if covariates.ndim == 1:
        probabilities = [single_logit(cov, parameters) for cov in covariates]
        lhoods = [resp*np.log(prob) + (1-resp)*np.log(1-prob) for prob, resp in zip(probabilities, responses)]
    else:
        probabilities = [general_logit(cov, parameters) for cov in covariates]
        lhoods = [resp*np.log(prob) + (1-resp)*np.log(1-prob) for prob, resp in zip(probabilities, responses)]
    return np.nansum(lhoods)

def logit_lpd(parameters, hyperparameters, responses, observed_covariates):
    '''
    Log posterior density function, calculated by calling the log prior and the log likelihood functions
    In the form necessary for targeting in the MH-MCMC algorithm function

    Arguments:
    responses - a list or numpy array length N of binary responses
    observed_covariates - a numpy matrix with N rows of covariate observations and M columns of covariate types
    parameters - a list of M+1 parameter values
    hyperparameters - a list of standard deviations to use in the Normal prior for each parameter
    '''
    log_prior = logit_lpl(parameters, hyperparameters)
    log_lik = logit_ll(parameters, responses, observed_covariates)
    return log_prior + log_lik

def normal_proposal(parameter, stepsize=0.5):
    proposal = np.random.normal(loc=parameter, scale=stepsize)
    return proposal

def neg_llh(parameters, responses, observations):
    '''
    Negative log-likelihood function

    Arguments are as log_likelihood

    Returns the negative log likelihood
    '''
    llh = logit_ll(parameters, responses, observations)
    return -llh

def logistic(x):
    '''
    Logistic function for transforming real number onto [0,1]
    '''
    return 1/(1+np.exp(-x))

def inv_logistic(x):
    '''
    Inverse logistic function for transforming [0,1] to real line
    '''
    return np.log(x/(1-x))

def mod4_mh_mcmc(iterations, stepsizes, target, initial_params, param_names, hyperparams, observations, responses):
    '''
    Model 4 (stick-breaking) sampling algorithm

    Transforms a normal proposal on the real line to a proposal in [0,1] to reflect the parameter space

    Arguments:
    iterations - an integer number of iterations to perform
    stepsizes - a list of stepsizes for each parameter
    target - the target distribution
    initial_params - the initial state of the chain
    param_names - names for the parameters to form the data frame
    hyperparams - hyperparameters for the Beta prior distributions
    observations - an array of observed event counts
    responses - an array of binary response variables

    Returns a data frame of the sampled parameters
    '''
    num_params = len(initial_params)
    samples = np.zeros((iterations+1, num_params))
    samples[0,:] = initial_params
    current_params = initial_params
    acceptance_count = np.zeros(num_params)

    for iter in range(iterations):
        for idx, param in enumerate(current_params):
            # propose the same parameters, changing the appropriate parameter only
            # this is a messy method but I'd not realised that lists aren't immutable
            # propose in the proposal space, transform to the parameter space
            current_unconstrained = inv_logistic(param)
            proposed_unconstrained = current_unconstrained + np.random.normal(scale=abs(np.log(stepsizes[idx])))
            proposed_param = logistic(proposed_unconstrained)
            proposed_params = np.zeros(3)
            proposed_params[:idx] = current_params[:idx]
            proposed_params[idx] = proposed_param
            if idx != num_params-1:
                proposed_params[idx+1:] = current_params[idx+1:]

            # find the target log densities under the current and proposed parameters, and the acceptance ratio
            log_post_current = target(current_params, responses, observations, hyperparams)
            log_post_proposed = target(proposed_params, responses, observations, hyperparams)
            acceptance_ratio = log_post_proposed - log_post_current

            if acceptance_ratio > np.log(np.random.rand()):
                # if accepted, update the parameters
                current_params = proposed_params
                acceptance_count[idx] += 1
                samples[iter+1, idx] = proposed_param
            
            else:
                samples[iter+1, idx] = param

    for idx, param in enumerate(param_names):
        print(f'Proposal acceptance rate of {acceptance_count[idx]/iterations} for parameter {param}')

    return_frame = pd.DataFrame(samples, columns = param_names)
    return return_frame

def mod4_lpl(parameters, hyperparameters):
    '''
    Model 4 log prior likelihood
    '''
    lpl = 0
    for param, hypers in zip(parameters, hyperparameters):
        lpl += stats.beta.pdf(param, hypers[0], hypers[1])
    return lpl

# def mod4_prob(parameters, count):
#     '''
#     Model 4 probability

#     Finds the correct probability given a list of sampled parameters and an event count

#     Arguments:
#     parameters - a list of parameters of the form [pK, qK-1, ..., q1]
#     count - an event count
#     '''
#     if count > 1:
#         return np.prod(parameters[:-count+1])
#     else:
#         return np.prod(parameters)

def mod4_prob(parameters, count):
    return np.prod(parameters[:len(parameters) + 1 - count])

def mod4_llh(parameters, responses, observations):
    '''
    Model 4 log likelihood

    Arguments:
    parameters - a list of the form [pK, qK-1,..., q1]
    counts - an array of event counts
    fatalities - an array of binary fatality indicators
    '''
    llh = 0
    for fatal, count in zip(responses, observations):
        prob = mod4_prob(parameters, count)
        llh += fatal*np.log(prob) + (1-fatal)*np.log(1-prob)
    return llh

def mod4_lpd(parameters, responses, observations, hyperparameters):
    '''
    Model 4 log posterior density function

    Arguments are as for mod4_lpl and mod4_llh
    '''
    return mod4_lpl(parameters, hyperparameters) + mod4_llh(parameters, responses, observations)

def mod6_mh_mcmc(iterations, stepsize, target, initial_param, param_name, hyperparams, observations, responses):
    '''
    Model 6 sampling algorithm

    Transforms a normal proposal on the real line to a proposal in [0,1] to reflect the parameter space
    Specifically for use with the one dimension of this model

    Arguments:
    iterations - an integer number of iterations to perform
    stepsizes - the stepsize for the parameter
    target - the target distribution
    initial_param - the initial state of the chain
    param_name - the name of the parameter
    hyperparams - hyperparameters for the Beta prior distribution
    observations - an array of observed event counts
    responses - an array of binary response variables

    Returns a data frame of the sampled parameters
    '''
    samples = np.zeros(iterations+1)
    samples[0] = initial_param
    current_param = initial_param
    acceptance_count = 0

    for iter in range(iterations):
            # propose the same parameters, changing the appropriate parameter only
            # this is a messy method but I'd not realised that lists aren't immutable
            # propose in the proposal space, transform to the parameter space
        current_unconstrained = inv_logistic(current_param)
        proposed_unconstrained = current_unconstrained + np.random.normal(scale=abs(np.log(stepsize)))
        proposed_param = logistic(proposed_unconstrained)

            # find the target log densities under the current and proposed parameters, and the acceptance ratio
        log_post_current = target(current_param, responses, observations, hyperparams)
        log_post_proposed = target(proposed_param, responses, observations, hyperparams)
        acceptance_ratio = log_post_proposed - log_post_current

        if acceptance_ratio > np.log(np.random.rand()):
                # if accepted, update the parameters
            current_param = proposed_param
            acceptance_count += 1
            samples[iter+1] = proposed_param
            
        else:
            samples[iter+1] = current_param

    print(f'Proposal acceptance rate of {acceptance_count/iterations} for parameter {param_name[0]}')

    return_frame = pd.DataFrame(samples, columns = param_name)
    return return_frame

def mod6_lpl(parameter, hyperparameters):
    return stats.beta.logpdf(parameter, hyperparameters[0], hyperparameters[1])

def mod6_llh(parameter, responses, observations):
    probabilities = [1-np.exp(count*np.log(1-parameter)) for count in observations]
    log_lhoods = [fatal * np.log(prob) + (1-fatal) * np.log(1-prob) for prob, fatal in zip(probabilities, responses)]
    return np.nansum(log_lhoods)

def mod6_lpd(parameter, responses, covariates, hyperparameters):
    return mod6_lpl(parameter, hyperparameters) + mod6_llh(parameter, responses, covariates)