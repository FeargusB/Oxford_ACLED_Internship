import numpy as np
import pandas as pd


def general_mh_mcmc(target_dist, initial_state, param_names, proposal_dists, stepsizes, iterations=10000, bounds=None, *args):
    '''
    A Metropolis-Hastings algorithm to approximate the parameters of a target distribution
    Currently assumes a symmetric proposal distribution for simplicity

    Arguments:
    target_dist - a function representing a log density which is to be approximated
    initial_state - an array or list of the state with which to initialise the chain
    parameter_names - a list of string parameter names
    proposal_dists - a list of functions, representing proposal distributions, taking the current parameter value and a stepsize as arguments
    stepsizes - a list of stepsizes for the proposal distributions
    iterations - an integer number of iterations to perform

    Returns:
    a pandas data frame with columns of iterations+1 sampled parameter values from the target distribution
    '''

    ## INITIAL SETUP

    # find the number of parameters
    num_params = len(param_names)

    # create a matrix to store the sampled values
    params_chain = np.zeros((iterations+1, num_params))
    params_chain[0,:] = initial_state

    # initialise parameters and acceptance rates
    current_state = initial_state
    param_acceptance_count = np.zeros(num_params)

    ## MAIN CHAIN SECTION - NO BOUNDS ON POSTERIOR

    # perform the required number of iterations
    if bounds == None:
        for i in range(iterations):
            # update the parameters sequentially
            for idx, state in enumerate(current_state):
                # propose the same parameters, changing the appropriate parameter only
                # this is a messy method but I'd not realised that lists aren't immutable
                new_state = proposal_dists[idx](state, stepsizes[idx])
                proposed_state = np.zeros(num_params)
                proposed_state[:idx] = current_state[:idx]
                proposed_state[idx] = new_state
                if idx != num_params-1:
                    proposed_state[idx+1:] = current_state[idx+1:]

                # find the target log densities under the current and proposed parameters, and the acceptance ratio
                log_post_current = target_dist(current_state, *args)
                log_post_proposed = target_dist(proposed_state, *args)
                acceptance_ratio = log_post_proposed - log_post_current

                if acceptance_ratio > np.log(np.random.rand()):
                    # if accepted, record the new parameter and update the parameters
                    params_chain[i+1, idx] = new_state
                    current_state = proposed_state
                    param_acceptance_count[idx] += 1

                else:
                    params_chain[i+1, idx] = state

    ## MAIN CHAIN SECTION - BOUNDS ON POSTERIOR

    if bounds != None:
        for i in range(iterations):
            # update the parameters sequentially
            for idx, state in enumerate(current_state):
                # propose the same parameters, changing the appropriate parameter only
                # this is a messy method but I'd not realised that lists aren't immutable
                new_state = proposal_dists[idx](state, stepsizes[idx])
                if new_state < bounds[idx][0] or new_state > bounds[idx][1]:
                    # proposal is outside the bounds so reject it, record previous state
                    params_chain[i+1, idx] = state
                else:
                    proposed_state = np.zeros(num_params)
                    proposed_state[:idx] = current_state[:idx]
                    proposed_state[idx] = new_state
                    if idx != num_params-1:
                        proposed_state[idx+1:] = current_state[idx+1:]

                    # find the target log densities under the current and proposed parameters, and the acceptance ratio
                    log_post_current = target_dist(current_state, *args)
                    log_post_proposed = target_dist(proposed_state, *args)
                    acceptance_ratio = log_post_proposed - log_post_current

                    if acceptance_ratio > np.log(np.random.rand()):
                        # if accepted, record the new parameter and update the parameters
                        params_chain[i+1, idx] = new_state
                        current_state = proposed_state
                        param_acceptance_count[idx] += 1

                    else:
                        params_chain[i+1, idx] = state
    
    for idx, param in enumerate(param_names):
        print(f'Proposal acceptance rate of {param_acceptance_count[idx]/iterations} for parameter {param}')

    params_chain_frame = pd.DataFrame(params_chain, columns = param_names)

    return params_chain_frame