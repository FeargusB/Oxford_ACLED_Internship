import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gammafunc
from scipy.special import logsumexp
from general_model import temporal_aggregation, filter_data_by_dict, stan_sample, bin_data, model5_coeffs, model5_observations
from mcmc_functions import logit_ll, logit_lpl

def marginal_likelihood(method, model, samples, data,):

    return None

def chib_marginal_likelihood(samples, prior_lhood, hyperparameters, likelihood, responses, observations,
                             bins=50, target_proportion=0.01, proportion_diagnostic=True, diagnostic_plot=False):
    '''
    Adapted Chib's method marginal likelihood estimation function

    Uses an estimate of the posterior density at a point close to the maximum a posteriori to find the marginal likelihood
    '''
    if samples.ndim == 1:
        max_ap = single_map(samples, bins=bins)
        log_posterior_density = np.log(univariate_post_density(max_ap, samples, target_proportion))
        if diagnostic_plot:
            if diagnostic_plot['proportion']:
                limits = diagnostic_plot['limits']
                length = diagnostic_plot['length']
                proportions = np.linspace(limits[0], limits[1], length)
                densities = []
                for prop in proportions:
                    densities.append(np.log(univariate_post_density(max_ap, samples, prop)))
                fig, ax = plt.subplots()
                ax.plot(proportions, densities)
                ax.set(title='Effect of proportion on density estimate',
                    xlabel='Aimed proportion',
                    ylabel='Estimated log posterior density at the MAP')
                plt.show()
    else:
        max_ap = joint_map(samples, bins=bins)
        log_posterior_density = np.log(multivariate_post_density(max_ap, samples, target_proportion, proportion_diagnostic=proportion_diagnostic))
        if diagnostic_plot:
            if diagnostic_plot['proportion']:
                limits = diagnostic_plot['limits']
                length = diagnostic_plot['length']
                proportions = np.linspace(limits[0], limits[1], length)
                densities = []
                for prop in proportions:
                    densities.append(np.log(multivariate_post_density(max_ap, samples, prop, proportion_diagnostic=False)))
                fig, ax = plt.subplots()
                ax.plot(proportions, densities)
                ax.set(title='Effect of proportion on density estimate',
                    xlabel='Aimed proportion',
                    ylabel='Estimated log posterior density at the MAP')
                plt.show()

    log_prior_density = prior_lhood(max_ap, hyperparameters)
    log_likelihood = likelihood(max_ap, responses, observations)

    if diagnostic_plot:
        limits = diagnostic_plot['limits']
        length = diagnostic_plot['length']
        proportions = np.linspace(limits[0], limits[1], length)
        marg_liks = []
        for prop in proportions:
            marg_liks.append(chib_marginal_likelihood(samples, prior_lhood, hyperparameters, likelihood, responses, observations,
                                                      bins, prop, proportion_diagnostic=False, diagnostic_plot=False))
        fig, ax = plt.subplots()
        ax.plot(proportions, marg_liks)
        ax.axvline(x=target_proportion, linestyle='--', label='target_proportion', c='r')
        ax.set(title='Effect of proportion on marginal likelihood estimate',
               xlabel='Aimed proportion in posterior density estimate',
               ylabel='Estimated log marginal likelihood')
    
    return log_prior_density + log_likelihood - log_posterior_density

def logistic_chib_marginal_likelihood(samples, data, filter_dict, covariates_dict,
                                      prior_lhood=None, likelihood=None, hyperparameters=None,
                                      bins=50, target_proportion=0.01, proportion_diagnostic=True, diagnostic_plot=False, ignore_zero=False):
    '''
    Adapted chib marginal likelihood function for the general logistic model

    Arguments:
    samples - a DATA FRAME of sampled parameters which has the output column names from model 5 sampling
    prior_lhood - prior log likelihood function (can really hardcode this in I feel, set as None, if None, set as logistic kind of thing)
    hyperparameters - hyperparameters for the prior distributions
    likelihood - a prior likelihood function (would it maybe not be better for this to be hardcoded in since it's known??)
    data - a data frame of the conflict data, used for finding the covariates and responses for the likelihood
    filter_dict - a filter dictionary containing information on ADM2_name and EVENT_TYPE
    covariates_dict - a standard format covariates dictionary (see model 5 functionality for further details)
    '''
    selected_covariates = model5_coeffs(covariates_dict)
    selected_covariates.insert(0, 'alpha')
    samples = samples[selected_covariates].values
    print(samples.shape)
    district, event_type = filter_dict['ADM2_name'], filter_dict['EVENT_TYPE']
    covariates, responses = model5_observations(data, district, event_type, covariates_dict, stan_code=False, ignore_zero=ignore_zero)
    if not hyperparameters:
        hyperparameters = [[0,1]] * 8
    if not prior_lhood:
        prior_lhood = logit_lpl
    if not likelihood:
        likelihood = logit_ll
    return chib_marginal_likelihood(samples, prior_lhood, hyperparameters, likelihood,
                                    responses=responses, observations=covariates, bins=bins, target_proportion=target_proportion,
                                    proportion_diagnostic=proportion_diagnostic, diagnostic_plot=diagnostic_plot)

def harmonic_mean_marginal_likelihood(samples, likelihood, responses, covariates):
    '''
    The Harmonic Mean marginal likelihood estimator, as suggested by Newton and Raftery and others

    Arguments:
    samples - sampled parameters, as a numpy array
    likelihood - likelihood function
    responses - list or numpy array of responses, is converted to numpy array
    covariates - list or numpy array of covariates, is converted to numpy array
    '''
    num_samples = len(samples)
    # find the negative log likelihood of each sample
    lik_list = [-likelihood(sample, np.array(responses), np.array(covariates)) for sample in samples]
    # find the marginal likelihood using the harmonic mean method
    marg_lik = np.log(num_samples) - logsumexp(lik_list)
    return marg_lik

def nr4_marginal_likelihood(samples, likelihood, responses, covariates, delta=0.1, initial_value=-100, error=1e-2, convergence_diagnostic=False):
    '''
    The fourth marginal likelihood estimation iteration method, as provided by Newton and Raftery

    Arguments:
    samples - a numpy array of posterior samples
    likelihood - a likelihood function with arguments in order sample, responses, covariates
    responses - a list or numpy array, converted to a numpy array
    covariates - a list or numpy array, converted to a numpy array
    delta - a tuning parameter, set to 0.1 by default
    initial_value - the initial value with which to begin the iteration. experimentation suggests a lower value converges faster, hence default -100
    error - the iteration stopping criterion, iteration will stop when the marginal likelihood has stabilised within this value
    convergence_diagnostic - boolean, if True, an iteration-by-iteration update is printed
    '''
    marg_lik_estimates = [initial_value]

    # find unchanging parameters in the marginal likelihood estimate
    m = len(samples)
    log_likelihoods = [likelihood(sample, np.array(responses), np.array(covariates)) for sample in samples]

    # first term of the fraction
    term1 = np.log(delta) + np.log(m) - np.log(1-delta)

    # second term of the fraction
    term2_logs = [ll - logsumexp([np.log(delta) + initial_value, np.log(1-delta) + ll]) for ll in log_likelihoods]
    term2 = logsumexp(term2_logs)

    # third term of the fraction
    term3 = term1 + initial_value

    # fourth term of the fraction
    term4_logs = [-logsumexp([np.log(delta) + initial_value, np.log(1-delta) + ll]) for ll in log_likelihoods]
    term4 = logsumexp(term4_logs)

    # find the numerator and the denominator of the fraction, hence find new marginal likelihood estimate
    numerator = logsumexp([term1, term2])
    denominator = logsumexp([term3, term4])
    marg_lik_estimates.append(numerator-denominator)

    while abs(marg_lik_estimates[-2] - marg_lik_estimates[-1]) > error:
        # find the previous marginal likelihood estimate
        prev_iter = marg_lik_estimates[-1]

        # second term of the fraction
        term2_logs = [ll - logsumexp([np.log(delta) + prev_iter, np.log(1-delta) + ll]) for ll in log_likelihoods]
        term2 = logsumexp(term2_logs)

        # third term of the fraction
        term3 = term1 + prev_iter

        # fourth term of the fraction
        term4_logs = [-logsumexp([np.log(delta) + prev_iter, np.log(1-delta) + ll]) for ll in log_likelihoods]
        term4 = logsumexp(term4_logs)

        # numerator and denominator, hence the new marginal likelihood estimate
        numerator = logsumexp([term1, term2])
        denominator = logsumexp([term3, term4])
        marg_lik_estimates.append(numerator-denominator)

        # print the convergence diagnostic if required
        if convergence_diagnostic:
            print(f'Estimate in iteration {len(marg_lik_estimates)-1} of {marg_lik_estimates[-1]}')

    # return the most recent marginal likelihood estimate
    return marg_lik_estimates[-1]

def single_map(samples, bins=50):
    '''
    Univariate maximum a posteriori estimate

    Uses a single dimension histogram to estimate the maximum a posteriori given some parameter samples

    Despite potential issues in the use of this method, these should be of little concern as all that is sought for the adapted Chib's method is an area of high density
    '''
    hist, bins = np.histogram(samples, bins=bins, density=True)
    map_bin_index = np.argmax(hist)
    map_estimate = (bins[map_bin_index] + bins[map_bin_index + 1]) / 2
    return map_estimate

def joint_map(samples, bins=50):
    '''
    Multivariate maximum a posteriori estimate

    Uses a multidimenstional histog
    '''
    num_params = samples.shape[1]
    if num_params > 5:
        bins = int((5*1e8) ** (1/num_params))
    hist, edges = np.histogramdd(samples, bins=bins)
    max_indices = np.unravel_index(np.argmax(hist), hist.shape)
    map_estimate = [(edges[i][max_indices[i]] + edges[i][max_indices[i] + 1])/2 for i in range(len(max_indices))]
    return map_estimate

def univariate_post_density(max_ap, samples, target_proportion):
    '''
    Univariate posterior density function
    '''
    distances = abs(samples - max_ap)
    num_required = int(len(samples) * target_proportion)
    closest_indices = np.argsort(distances)[:num_required]
    closest_samples = samples[closest_indices]
    width = np.max(closest_samples) - np.min(closest_samples)
    actual_proportion = len(closest_samples)/len(samples)
    return actual_proportion/width

def multivariate_post_density(max_ap, samples, target_proportion, proportion_diagnostic=True):
    '''
    Multivariate posterior density estimation function
    '''
    dimension = samples.shape[1]

    distances = np.linalg.norm(samples-max_ap, axis=1)
    num_closest = int(len(samples)*target_proportion)
    closest_indices = np.argsort(distances)[:num_closest]
    closest_samples = samples[closest_indices]

    distances_from_map = np.linalg.norm(closest_samples-max_ap, axis=1)
    radius = np.max(abs(distances_from_map))

    distances = np.linalg.norm(samples-max_ap, axis=1)
    samples_in = np.count_nonzero(distances <= radius)

    actual_proportion = samples_in/len(samples)

    if proportion_diagnostic:
        print(f'aimed for {100*target_proportion}% of samples\nachieved  {100*actual_proportion}%')

    volume = hypersphere_volume(radius, dimension)

    return actual_proportion/volume

def hypersphere_volume(radius, dimension):
    '''
    Volume of a hypersphere of a certain radius given the number of dimensions
    '''
    volume = np.pi**(dimension/2) * abs(radius)**dimension / gammafunc(dimension/2 + 1)
    return volume

def null_hypothesis_sample(data, data_filter, max_bin, stan_filepath, sample_size=10000):

    with open(stan_filepath) as f:
        stan_code = f.read()

    filtered_data = filter_data_by_dict(data, data_filter)
    agg_data = temporal_aggregation(filtered_data, 'WEEK')

    bins = [i for i in range(max_bin + 1)]
    bins.append(float(('inf')))
    agg_data = bin_data(agg_data, bins=bins)
    fatality_flag = agg_data['FATALITY_FLAG'].values
    event_bins = agg_data['BINS'].values.astype(int)
    num_obs = len(fatality_flag)
    max_events = max_bin

    model_data = {'fatality_flag':fatality_flag,
                  'event_bins':event_bins,
                  'num_obs':num_obs,
                  'max_events':max_events}
    print('Sampling using Stan')

    return stan_sample(model_data, stan_code, size_out=sample_size)

def mc_marginal_likelihood(prior_dist, hyperparameters, likelihood, responses, covariates, num_samples=int(1e4)):
    samples = prior_dist(hyperparameters, size=num_samples)
    lhoods = [likelihood(sample, responses, covariates) for sample in samples]
    return logsumexp(lhoods)

def mod1_prior_sample(hyperparameters, size=int(1e4)):
    alpha_samples = np.random.normal(loc=hyperparameters[0], scale=hyperparameters[1], size=size)
    beta_samples = np.random.normal(loc=hyperparameters[0], scale=hyperparameters[1], size=size)
    samples = [[alpha, beta] for alpha, beta in zip(alpha_samples, beta_samples)]
    return np.array(samples)

def mod6_prior_sample(hyperparameters, size=int(1e4)):
    p_samples = np.random.beta(hyperparameters[0], hyperparameters[1], size=size)
    return np.array(p_samples)

def mod4_prior_sample(hyperparameters, max_bin=3, size=int(1e4)):
    samples = []
    for _ in range(max_bin):
        samples.append(np.random.beta(hyperparameters[0], hyperparameters[1], size=size))
    samples = np.array(samples)
    return np.rot90(samples)
