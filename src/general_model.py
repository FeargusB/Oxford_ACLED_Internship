import pandas as pd
import numpy as np
import arviz as az
import stan
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from mcmc_functions import *

def model_sample(model_type, data, data_filter, sample_size=10000, temp_agg='WEEK',
                 parameter_info=None, stan_filepath=None, max_bin=None, covariates_dict=None, ignore_zero=False, simulated_data=None, stan_seed=1,
                 null_hyperparameters=[1,1], bin_size=0.05):
    '''
    Generalised model sampling function

    Samples from the provided model using the given data and filter, and any further provided key word arguments. Can use Metropolis Hastings or Stan for sampling

    Arguments:
    model_type - model selector, currently 1-6 or 'null' (model 6). otherwise, function returns None
    data - the raw data frame as read from the data file
    data_filter - a dictionary containing information about how to filter the data
    sample_size - the number of posterior samples to return
    temp_agg - the temporal aggregation level
    parameter_info - a list of parameter names, hyperparameters and stepsizes for the MH MCMC algorithm. Either this or a stan filepath must be provided
    stan_filepath - a path to a Stan file. MH MCMC will be used only if this is False/None/0
    max_bin - a maximum binning number for models 3 and 4
    covariates_dict - a dictionary of covariate usage with boolean values for model 5
    ignore_zero - provides an option to ignore weeks with zero events in the analysis
    simulated_data - dictionary of simulated 'responses' and 'covariates'
    stan_seed - a random seed, primarily for Stan modelling but is used for Metropolis Hastings as well
    null_hyperparameters - a list of the two hyperparameters for the beta prior for the null model

    Returns:
    - when using Stan, a tuple (model, samples) of stan.fit.Fit and a data frame
    - when using Metropolis Hastings, a data frame of samples
    '''
    if stan_filepath == None and parameter_info == None and model_type not in [0, 7, 'null']:
        raise ValueError('Either a stan filepath or parameter information must be passed into the function')
    # retrieve the stan code if appropriate
    if stan_filepath:
        with open(stan_filepath) as f:
            stan_code = f.read()
    else:
        stan_code = None

    if max_bin == 'auto':
        max_bin = auto_max_bin(data, data_filter, temp_agg=temp_agg, bin_size=bin_size)

    if not simulated_data and model_type not in [0, 7, 'null']:
        # perform data manipulation
        check_data_columns(data, filter_dict=data_filter, temp_agg=temp_agg, extra_columns=['RAW_COUNTS', 'FATALITIES'])
        filtered_data = filter_data_by_dict(data, data_filter)
        agg_data = temporal_aggregation(filtered_data, temp_agg)

        # ignore periods with zero events if appropriate
        if ignore_zero:
            agg_data = agg_data.loc[agg_data['RAW_COUNTS'] != 0]

        responses, covariates = model_observations(model_type, data, data_filter, temp_agg, stan_filepath, max_bin, covariates_dict, ignore_zero)
    
    if simulated_data:
        responses = simulated_data['responses']
        covariates = simulated_data['covariates']

    if model_type == 1:
        if stan_code:
            data_dict = {'fatality_flag':responses,
                         'event_counts':covariates,
                         'num_obs':len(covariates)}
            print('Sampling using Stan, returning (stan.fit.Fit, data frame)')
            return stan_sample(data_dict, stan_code, sample_size, stan_seed=stan_seed)
        else:
            initial_params = list(minimize(neg_llh, [0, 0], (responses, covariates), method='L-BFGS-B').x)
            parameter_names = parameter_info['names']
            hyperparameters = parameter_info['hyperparameters']
            stepsizes = parameter_info['stepsizes']
            proposal_dists = [normal_proposal, normal_proposal]
            print('Sampling using Metropolis Hastings MCMC, returning data frame')
            np.random.seed(stan_seed)
            full_chain = logit_mh_mcmc(responses, covariates,
                                       parameter_names, initial_params, stepsizes, hyperparameters,
                                       proposal_dists, logit_lpd, iterations=sample_size)
            return full_chain

    if model_type == 2:
        # make the required adjustments to the data frame, then pull the responses and covariates
        agg_data['PREV_WEEK'] = (agg_data['FATALITY_FLAG'].shift(1) > 0).astype(int)
        responses = agg_data['FATALITY_FLAG'].values[1:]
        # use either stan or mh-mcmc as appropriate
        if stan_code:
            event_counts = covariates[:,0]
            print(f'{len(event_counts)} event count weeks with a maximum of {np.max(event_counts)}')
            prev_fatalities = covariates[:,1]
            print(f'maximum previous fatality indicator value of {np.max(prev_fatalities)}')
            data_dict = {'fatality_flag':responses,
                         'event_counts':event_counts,
                         'prev_fatality_flag':prev_fatalities,
                         'num_obs':len(prev_fatalities)}
            print('Sampling using Stan, returning (stan.fit.Fit, data frame)')
            return stan_sample(data_dict, stan_code, sample_size, stan_seed=stan_seed)
        else:
            covariates = agg_data[['RAW_COUNTS', 'PREV_WEEK']].values[1:]
            initial_params = list(minimize(neg_llh, [0, 0, 0], (responses, covariates), method='L-BFGS-B').x)
            parameter_names = parameter_info['names']
            hyperparameters = parameter_info['hyperparameters']
            stepsizes = parameter_info['stepsizes']
            proposal_dists = [normal_proposal, normal_proposal, normal_proposal]
            print('Sampling using Metropolis Hastings MCMC, returning data frame')
            np.random.seed(stan_seed)
            raw_chain = logit_mh_mcmc(responses, covariates,
                                       parameter_names, initial_params, stepsizes, hyperparameters,
                                       proposal_dists, logit_lpd, iterations=sample_size)
            return raw_chain
    
    if model_type == 3:
        if max_bin:
            print('Model 3 functionality still in progress')
            return None
        else:
            raise ValueError('Must provide binning information for model 3')
    
    if model_type == 4:
        parameter_names = ['pK']
        parameter_names.extend([f'q.{k}' for k in reversed(range(1, max_bin))])
        print(f'Discretisation and Ordering using {max_bin} bins with parameter names {parameter_names}')
        # use stan or mh-mcmc as appropriate
        if stan_code:
            max_events = max_bin
            model_data = {'fatality_flag':responses,
                          'event_bins':covariates,
                          'num_obs':len(responses),
                          'max_events':max_events}
            print('Sampling using Stan, returning (stan.fit.Fit, data frame)')
            return stan_sample(model_data, stan_code, size_out=sample_size, stan_seed=stan_seed)
        else:
            hyperparameters = parameter_info['hyperparameters']
            stepsizes = parameter_info['stepsizes']
            initial_params = [0.5] * len(parameter_names)
            print('Sampling using Metropolis-Hastings MCMC, returning data frame')
            np.random.seed(stan_seed)
            full_chain = mod4_mh_mcmc(iterations=sample_size, stepsizes=stepsizes, target=mod4_lpd,
                                      initial_params=initial_params, param_names=parameter_names, hyperparams=hyperparameters,
                                      observations=covariates, responses=responses)
            return full_chain
        
    if model_type == 5:
        # use the covariate dictionary and the data filter to use the more specific function
        if covariates_dict:
            district = data_filter['ADM2_name']
            event_type = data_filter['EVENT_TYPE']
            return model5_sample(data, district, event_type, covariates_dict, sample_size, stan_filepath, ignore_zero=ignore_zero, stan_seed=stan_seed)
        else:
            raise ValueError('Further information must be provided for model 5')
        
    if model_type == 6:
        if stan_code:
            num_obs = len(responses)
            model_data = {'fatality_flag':responses,
                          'event_counts':covariates,
                          'num_obs':num_obs}
            print('Sampling using Stan, returning (stan.fit.Fit, data frame)')
            return stan_sample(model_data, stan_code, size_out=sample_size, stan_seed=stan_seed)
        else:
            parameter_name = parameter_info['names']
            hyperparameters = parameter_info['hyperparameters']
            stepsize = parameter_info['stepsizes']
            initial_param = 0.5
            print('Sampling using Metropolis-Hastings MCMC, returning data frame')
            np.random.seed(stan_seed)
            full_chain = mod6_mh_mcmc(iterations=sample_size, stepsize=stepsize, target=mod6_lpd,
                                      initial_param=initial_param, param_name=parameter_name, hyperparams=hyperparameters,
                                      observations=covariates, responses=responses)
        return full_chain
    
    if model_type == 7 or model_type == 'null' or model_type == 0:
        print('Sampling directly from the Beta distribution, returning data frame')
        check_data_columns(data, filter_dict=data_filter, temp_agg=temp_agg, extra_columns=['RAW_COUNTS', 'FATALITIES'])
        filtered_data = filter_data_by_dict(data, data_filter)
        agg_data = temporal_aggregation(filtered_data, temp_agg)
        zero_fatal = len(agg_data.loc[agg_data['RAW_COUNTS'] == 1].loc[agg_data['FATALITIES'] == 0])
        one_fatal = len(agg_data.loc[agg_data['RAW_COUNTS'] == 1].loc[agg_data['FATALITIES'] > 0])
        alpha, beta = null_hyperparameters[0] + one_fatal, null_hyperparameters[1] + zero_fatal
        np.random.seed(stan_seed)
        posterior_samples = np.random.beta(alpha, beta, size=sample_size)
        return pd.DataFrame({'p.1':posterior_samples})
        
    return None

def auto_max_bin(data, data_filter, temp_agg='WEEK', bin_size=0.05):
    '''
    Automatic maximum bin finding function for model 4

    Aggregates the weeks with the most events together until the highest bin has bin_size proportion of the total count

    Arguments:
    data - the data as passed into model_sample
    data_filter - the filter dictionary as passed into model_sample
    temp_agg - as for model_sample
    bin_size - the minimum proportion of all observations to be in the maximum bin

    Returns an integer number to be used for max_bin in model_sample
    '''
    check_data_columns(data, filter_dict=data_filter, temp_agg=temp_agg, extra_columns=['RAW_COUNTS', 'FATALITIES'])
    filtered_data = filter_data_by_dict(data, data_filter)
    agg_data = temporal_aggregation(filtered_data, temp_agg)
    agg_data = agg_data.loc[agg_data['RAW_COUNTS'] > 0]

    current_event = np.max(agg_data['RAW_COUNTS'].values)
    total_weeks = len(agg_data)
    required_total = total_weeks * bin_size
    counted_events = len(agg_data.loc[agg_data['RAW_COUNTS'] == current_event])

    while counted_events <= required_total:
        current_event -= 1
        counted_events += len(agg_data.loc[agg_data['RAW_COUNTS'] == current_event])
        
    return int(current_event)

def model_observations(model_type, data, data_filter, temp_agg='WEEK', stan_filepath=None, max_bin=None, covariates_dict=None, ignore_zero=False):
    '''
    Model Observations function

    Given a model type and appropriate associated information, return the required responses and covariates

    Arguments:
    model_type - model selector, currently 1-6 or 'null' (model 6). otherwise, function returns None
    data - the raw data frame as read from the data file
    data_filter - a dictionary containing information about how to filter the data
    temp_agg - the temporal aggregation level
    stan_filepath - essentially used as a toggle for how to find the model 5 covariates
    max_bin - a maximum binning number for models 3 and 4
    covariates_dict - a dictionary of covariate usage with boolean values for model 5
    ignore_zero - provides an option to ignore weeks with zero events in the analysis

    Returns a tuple of (responses, covariates) for the appropriate model
    '''

    # check the data frame
    check_data_columns(data, filter_dict=data_filter, temp_agg=temp_agg, extra_columns=['RAW_COUNTS', 'FATALITIES'])
    filtered_data = filter_data_by_dict(data, data_filter)
    agg_data = temporal_aggregation(filtered_data, temp_agg)

    # ignore weeks of zero events as appropriate
    if ignore_zero:
        agg_data = agg_data.loc[agg_data['RAW_COUNTS'] != 0]

    if model_type == 1:
        responses = agg_data['FATALITY_FLAG'].values
        covariates = agg_data['RAW_COUNTS'].values
        return responses, covariates
    
    if model_type == 2:
        agg_data['PREV_WEEK'] = (agg_data['FATALITY_FLAG'].shift(1) > 0).astype(int)
        responses = agg_data['FATALITY_FLAG'].values[1:]
        covariates = agg_data[['RAW_COUNTS', 'PREV_WEEK']].values[1:]
        return responses, covariates
    
    if model_type == 4:
        if max_bin:
            bins = [i for i in range(max_bin + 1)]
            bins.append(float('inf'))
            agg_data = bin_data(agg_data, bins=bins)
            fatality_flag = agg_data['FATALITY_FLAG'].values
            event_bins = agg_data['BINS'].values.astype(int)
        else:
            raise ValueError('Must provide binning information for model 4')
        responses = [flag for flag, cov in zip(fatality_flag, event_bins) if cov > 0]
        covariates = [cov for cov in event_bins if cov > 0]
        return responses, covariates
    
    if model_type == 5:
        district, event_type = data_filter['ADM2_name'], data_filter['EVENT_TYPE']
        return model5_observations(data, district, event_type, covariates_dict, stan_filepath)

    if model_type == 6 or model_type == 'null':
        fatality_flag = agg_data['FATALITY_FLAG'].values
        event_counts = agg_data['RAW_COUNTS'].values
        responses = [flag for flag, cov in zip(fatality_flag, event_counts) if cov > 0]
        covariates = [cov for cov in event_counts if cov > 0]
        return responses, covariates
    
    return None

def model5_sample(data, district, event_type, covariates_dict, sample_size=10000, stan_filepath=None, ignore_zero=False, stan_seed=1):
    '''
    Sampling function for model(s) 5 - the generalised inner/outer modelling

    Uses the covariates dictionary (values should be boolean) to select the covariates and coefficients for the model
    Could be integrated into the model_sample function with some adjustment in that function
    Currently uses the Metropolis Hastings algorithm to sample

    Arguments:
    data - a pandas data frame with the required columns
    district - a string corresponding to the name of a level 2 administrative region (district) in Bangladesh
    event_type - one of the six ACLED conflict types
    covariates_dict - a dictionary with boolean values turning covariates specified by the keys on and off
    sample_size - length of returned sample
    stan_filepath - path to the stan file
    ignore_zero - toggle for ignoring weeks with zero events
    stan_seed - random seed for Stan and Metropolis Hastings sampling

    Returns - see model_sample
    '''
    # check the required columns
    check_data_columns(data, extra_columns=['ADMIN1', 'ADM2_name', 'WEEK', 'RAW_COUNTS', 'FATALITIES'])

    # find the parameter names, add alpha as a parameter
    parameter_names = model5_coeffs(covariates_dict)
    parameter_names.insert(0, 'alpha')
    
    # use stan or mh-mcmc as appropriate
    if stan_filepath:
        with open(stan_filepath) as f:
            stan_code = f.read()
        print('Sampling using Stan, returning (stan.fit.Fit, data frame)')
        num_covs = len(covariates_dict.keys())
        # find the covariates and responses
        covariates, responses = model5_observations(data, district, event_type, covariates_dict, stan_code=True, ignore_zero=ignore_zero)
        num_obs = len(responses)
        inclusion = [int(value) for value in covariates_dict.values()]
        model_data = {'num_obs':num_obs,
                      'num_covs':num_covs,
                      'observations':covariates,
                      'inclusion':inclusion,
                      'responses':responses}
        return stan_sample(model_data, stan_code, size_out=sample_size, stan_seed=stan_seed)
    else:
        print('Sampling using Metropolis Hastings MCMC, returning data frame')
        # find covariates, responses and stepsizes
        covariates, responses = model5_observations(data, district, event_type, covariates_dict, ignore_zero=ignore_zero)
        parameter_stepsizes = model5_steps(covariates_dict)
        parameter_stepsizes.insert(0, 0.7)
        # normal priors and proposals
        hyperparameters = [[0,1]] * len(parameter_names)
        proposal_dists = [normal_proposal] * len(parameter_names)
        # start from the origin as maxmizing the lkelihood is not effective in some situations
        initial_parameters = [0] * len(parameter_names)

        samples = logit_mh_mcmc(responses, covariates,
                                parameter_names, initial_parameters, parameter_stepsizes, hyperparameters,
                                proposal_dists, logit_lpd, iterations=sample_size)

        return samples

def model5_data_prepare(data, district, event_type, ignore_zero=False):
    '''
    Model 5 data preparation function

    Prepares a data frame to streamline observed covariate selection for the sampling algorithm

    Arguments:
    data - a data frame of event and fatality data
    district - a district of Bangladesh on which the model is centred
    event_type - the event type(s) to be selected

    Returns a data frame grouped by week of the weekly totals of events of the chosen type and a fatality indicator for the district and the surrounding division
    '''
    division = bangladesh_division_from_district(district)

    # find information on the district, rename as appropriate
    district_data = data.loc[data['ADM2_name'] == district].loc[data['EVENT_TYPE'] == event_type].groupby('WEEK', as_index=False)[['RAW_COUNTS', 'FATALITIES']].sum()
    district_data['BOOLEAN_FATAL'] = district_data['FATALITIES'].gt(0).astype('int')
    district_data = district_data.rename(columns={'BOOLEAN_FATAL':'inner_fatal', 'RAW_COUNTS':'inner_events'})
    district_data['inner_fatal_prev'] = (district_data['inner_fatal'].shift(1) > 0).astype(int)
    district_data['inner_events_prev'] = (district_data['inner_events'].shift(1))

    # find information on the division, rename and rejig as appropriate
    division_data = data.loc[data['ADMIN1'] == division].loc[data['ADM2_name'] != district].loc[data['EVENT_TYPE'] == event_type].groupby('WEEK', as_index=False)[['RAW_COUNTS', 'FATALITIES']].sum()
    division_data['BOOLEAN_FATAL'] = division_data['FATALITIES'].gt(0).astype(int)
    division_data = division_data.rename(columns={'BOOLEAN_FATAL':'outer_fatal', 'RAW_COUNTS':'outer_events'})
    division_data['outer_fatal_prev'] = (division_data['outer_fatal'].shift(1) > 0).astype(int)
    division_data['outer_events_prev'] = (division_data['outer_events'].shift(1))

    # merge and return the two data frames
    merged_data = pd.merge(district_data, division_data, on='WEEK')
    if ignore_zero:
        merged_data = merged_data.loc[merged_data['inner_events'] != 0]
    return merged_data

def model5_observations(data, district, event_type, covariates_dict, stan_code=False, ignore_zero=False):
    '''
    Model 5 observation finding function

    Given the chosen covariates, finds the correct observations to use to build the model. If Stan is being used, all covariates are needed

    Arguments:
    data - a data frame which is passed through the data preparation function
    district - a district to centre the model on
    event_type - a type of event to model
    covariates_dict - a dictionary with boolean values indicating the usage of each covariate
    stan_code - an essentially boolean indicator of whether Stan is being used

    Returns an array of observations. If stan is used, this is all covariates, regardless of the model
    '''
    # prepare the data with the correct columns
    prepared_data = model5_data_prepare(data, district, event_type, ignore_zero)
    if stan_code:
        # if stan is being used all covariates must be used as required by the stan file
        required_columns = [key for key in covariates_dict.keys()]
    else:
        # if using mh-mcmc need only the appropriate covariates, otherwise algorithm will not work
        required_columns = [key for key, value in covariates_dict.items() if value]
    # return covariates, responses in that order
    if len(required_columns) == 1:
        return prepared_data[required_columns[0]].values[1:], prepared_data['inner_fatal'].values[1:]
    else:
        return prepared_data[required_columns].values[1:], prepared_data['inner_fatal'].values[1:]

def model5_coeffs(covariates_dict):
    '''
    Coefficient naming function for the generalised logistic model

    Given a dictionary of considered covariates, returns the list of parameter names. Enables consistency across model comparison

    Arguments:
    covariates_dict - a dictionary values True or False of covariates being used by the model

    Returns a list of the coefficient names for the model
    '''
    coeff_dict = {'inner_events':'beta_.1',
                  'inner_events_prev':'beta_.2',
                  'inner_fatal_prev':'beta_.3',
                  'outer_events':'beta_.4',
                  'outer_events_prev':'beta_.5',
                  'outer_fatal':'beta_.6',
                  'outer_fatal_prev':'beta_.7'}
    coefficients = [value for key, value in coeff_dict.items() if covariates_dict[key]]
    return coefficients

def model5_steps(covariates_dict):
    '''
    Drawing stepsizes for the generalised logit model

    Given a dictionary of considered covariates, returns the list of suggested stepsizes. Stepsizes can be modified here for now

    Arguments:
    covariates_dict - a dictionary values True or False of covariates being used by the model

    Returns a list of the stepsizes for the model
    '''
    steps_dict = {'inner_events':0.7,
                  'inner_events_prev':0.7,
                  'inner_fatal_prev':1.0,
                  'outer_events':1.0,
                  'outer_events_prev':0.1,
                  'outer_fatal':0.1,
                  'outer_fatal_prev':0.5}
    stepsizes = [value for key, value in steps_dict.items() if covariates_dict[key]]
    return stepsizes

def logit_graph(alpha_list, beta_list, x_limits = [-1, 5], figsize=(15, 5), colour='b', fig=None, ax=None):
    '''
    Logistic graph plotting function, needs work, I need to consider more carefully how to display posterior information
    Could be the case that I end up only allowing use of this for models 1 and 2, or embedding it into a different function
    '''
    x_axis = np.linspace(x_limits[0], x_limits[1], 1000)
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)
    for alpha, beta in zip(alpha_list, beta_list):
        probs = single_logit(x_axis, [alpha, beta])
        ax.plot(x_axis, probs, c=colour, alpha=0.1)
    return fig, ax
    
def sampling_visualisation(sample, param_names=None, figsize=(10,10), pairplot=False, pdf_bins=100):
    '''
    Sampling visualisation function to plot trace plots and where appropriate autocorrelation and pairplots

    Arguments:
    sample - a data frame of posterior samples OR a stan.fit.Fit object
    param_names - a list of parameter names which correspond to columns of the data frame or stan fit object
    figsize - a size for the traceplots
    pairplot - a boolean toggle for whether to also plot a pairplot
    '''
    # Stan model object
    if type(sample) == stan.fit.Fit:
        # plot the trace using the model object
        az.plot_trace(sample, figsize=figsize, legend=True)
        # plot the pairplot using the provided parameter names
        if pairplot:
            if param_names:
                check_data_columns(sample.to_frame(), extra_columns=param_names)
                sns.pairplot(sample.to_frame()[param_names])
            else:
                raise ValueError('Parameter names required for a pairplot')

    # data frame of sampled parameters   
    else:
        if param_names:
            # check the parameter names, create a figure and axes
            check_data_columns(sample, extra_columns=param_names)
            num_params = len(param_names)
            if figsize == (10,10):
                figsize = (15, num_params*5)
            fig, ax = plt.subplots(num_params, 3, figsize=figsize)
            # plot all required trace and autocorrelation plots
            # the one parameter case must be handled separately
            if num_params == 1:
                # pdf using kernel density estimation
                sns.kdeplot(data=sample[param_names], ax=ax[0])
                param = param_names[0]
                ax[0].set(title=f'Approximate p.d.f. for parameter {param}')
                # trace plot
                ax[1].plot(sample[param])
                ax[1].set(title=f'Sampled {param} chain',
                               ylabel=f'{param}')                    
                # autocorrelation plot
                az.plot_autocorr(sample[param].values, ax=ax[2])
                ax[2].set(title=f'Autocorrelation plot for {param}')
            else:
                for idx, param in enumerate(param_names):
                    # pdf using kernel density estimation
                    sns.kdeplot(data=sample[param], ax=ax[idx, 0])
                    ax[idx, 0].set(title=f'Approximate p.d.f. for parameter {param}')
                    # trace plot
                    ax[idx, 1].plot(sample[param])
                    ax[idx, 1].set(title=f'Sampled {param} chain',
                                   ylabel=f'{param}')                    
                    # autocorrelation plot
                    az.plot_autocorr(sample[param].values, ax=ax[idx, 2])
                    ax[idx, 2].set(title=f'Autocorrelation plot for {param}')
            if pairplot:
                sns.pairplot(sample[param_names])
        else:
            raise ValueError('Parameter names required if sampling argument not a stan.fit.Fit object')
    return fig, ax

def print_ess(samples, param_names):
    '''
    Effective sample size display function

    Displays the effective sample size for sampled parameters

    Arguments:
    samples - a data frame of samples
    param_names - parameter names of which to find the effective sample size

    Returns None, prints the effective sample size of each of the sampled parameters as determined by the parameter names
    '''
    check_data_columns(samples, extra_columns=param_names)
    for param in param_names:
        ess = az.ess(samples[param].values)
        print(f'Effective sample size of {ess} for parameter {param}')
    return None

def stan_sample(data, code, size_out=10000, stan_seed=1):
    '''
    Stan sampling function

    Shorthand function to sample using Stan, given code and data.
    Uses an inbuilt parameters for everything but the number of samples returned.

    Arguments:
    data - a dictionary of data
    code - stan code which is compatible with the data
    size_out - the number of samples to be returned after burning and thinning

    Returns a tuple of a stan.fit.fit object and its corresponding data frame
    '''
    # initialise other parameters
    num_chains, num_warmup, num_thin, refresh, delta = 4, 2000, 10, 1000, 0.8
    num_samples = int(size_out*num_thin/num_chains)

    # build the model, sample from it
    model = stan.build(code, data=data, random_seed=stan_seed)
    samples = model.sample(num_samples = num_samples,
                           num_chains = num_chains,
                           num_warmup = num_warmup,
                           num_thin = num_thin,
                           refresh = 25000,
                           delta = delta)
    # return the samples for analysis and as a data frame 
    return samples, samples.to_frame()

def check_data_columns(data, filter_dict=None, temp_agg=None, extra_columns=None):
    '''
    Checks that the data, filter dictionary and temporal aggregation are compatible, raises errors otherwise

    Arguments:
    data - the data frame used in the general model function
    filter_dict - the filtering dictionary passed into the model_sample function
    temp_agg - the attempted temporal aggregation

    Returns None
    Raises a ValueError if any of the column names are not in the column names of the data frame
    '''
    # find the column names, make a note of the observation columns required
    column_names = list(data)

    # check the filtering dictionary
    if filter_dict:
        for key in filter_dict.keys():
            if key not in column_names:
                raise ValueError(f'Filtering column "{key}" not in column names of data frame')
        
    # check the temporal aggregation
    if temp_agg:
        if temp_agg not in column_names:
            raise ValueError(f'Temporal aggregation "{temp_agg}" not in column names of data frame')
    
    # check extra columns
    if extra_columns:
        for col in extra_columns:
            if col not in extra_columns:
                raise ValueError(f'Filtering column "{col}" not in column names of data frame')

    return None

def filter_data_by_dict(data, filter_dict):
    '''
    Use a dictionary to filter a data frame

    Arguments:
    data - a pandas data frame
    filter_dict - a dictionary with keys corresponding to column names of the data frame
                                and values corresponding to data values in those columns

    Returns a data frame filtered for where each key takes the corresponding value
    '''
    # create a mask based on the values of the dictionary and the data frame
    mask = pd.Series(True, index=data.index)

    for key, value in filter_dict.items():
        mask &= (data[key] == value)

    return data[mask]

def temporal_aggregation(data, temp_agg):
    '''
    Temporal aggregation of the data

    Arguments:
    data - a pandas data frame, should have a column with heading temp_agg
    temp_agg - a string column heading
    columns - columns to keep in the grouping, these get summed within each group

    Returns a data frame group appropriately with the specified columns summed, and a fatality indicator
    '''
    agg_data = data.groupby(temp_agg, as_index=False)[['RAW_COUNTS', 'FATALITIES']].sum()
    agg_data['FATALITY_FLAG'] = agg_data['FATALITIES'].gt(0).astype('int')
    return agg_data

def bin_data(data, bins, binning_column='RAW_COUNTS', new_column='BINS', bin_labels=None):
    '''
    Data binning function

    Arguments:
    data - a pandas data frame which should have passed through check_data_columns
    bins - a list of numbers defining intervals by which to bin the data
    binning_column - the column to be binned, default is 'RAW_COUNTS' as this is the intention of the function
    new_column - the name of the new column
    bin_labels - the values of the new column, default None which uses 0,1,...,n-1 (where there are n bins)
    '''
    if bin_labels:
        data[new_column] = pd.cut(data[binning_column], bins=bins, right=False, labels=bin_labels)
    else:
        data[new_column] = pd.cut(data[binning_column], bins=bins, right=False, labels=range(len(bins)-1))
    return data

def bangladesh_division_from_district(district):
    '''
    Finds the division (level 1 region) containing a given district (level 2 region)
    Raises an error if the supplied name does not exactly 

    Arguments:
    district - a string district of Bangladesh
    '''
    district = district.lower()
    if district in ['barguna', 'barisal', 'bhola', 'jhalokati', 'patuakhali', 'pirojpur']:
        return 'Barisal'
    elif district in ['bandarban', 'brahamanbaria', 'chandpur', 'chittagong', 'comilla',
                      "cox's bazar", 'feni', 'khagrachhari', 'lakshmipur', 'noakhali', 'rangamati']:
        return 'Chittagong'
    elif district in ['dhaka', 'faridpur', 'gazipur', 'gopalganj', 'kishoreganj',
                      'madaripur', 'manikganj', 'munshiganj', 'narayanganj', 'narsingdi',
                      'rajbari', 'shariatpur', 'tangail']:
        return 'Dhaka'
    elif district in ['bagerhat', 'chuadanga', 'jessore', 'jhenaidah', 'khulna',
                      'kushtia', 'magura', 'meherpur', 'narail', 'satkhira']:
        return 'Khulna'
    elif district in ['jamalpur', 'mymensingh', 'netrakona', 'sherpur']:
        return 'Mymensingh'
    elif district in ['bogra', 'joypurhat', 'naogaon', 'natore', 'nawabganj', 'pabna',
                      'rajshahi', 'sirajganj']:
        return 'Rajshahi'
    elif district in ['dinajpur', 'gaibandha', 'kurigram', 'lalmonirhat', 'nilphamari',
                      'panchagarh', 'rangpur', 'thakurgaon']:
        return 'Rangpur'
    elif district in ['habiganj', 'maulvibazar', 'sunamganj', 'sylhet']:
        return 'Sylhet'
    else:
        raise ValueError(f'{district} is not a recognised level 2 district of Bangladesh')
    
def temporally_augment(data):
    '''
    Add temporal information to a database with year and day columns
    Creates a date, month and week column and returns the augumented data frame

    Arguments:
    data - a pandas data frame with columns including year and day
    '''
    if 'YEAR' not in list(data) or 'DAY' not in list(data):
        raise ValueError('Not enough temporal information in the data frame; "YEAR" and "DAY" columns required')
    
    # add date, month and week columns
    data['DATE'] =  pd.to_datetime(data['YEAR'].astype(int).astype(str) + data['DAY'].astype(int).astype(str), format='%Y%j')
    data['MONTH'] = data['DATE'].dt.to_period ('M')
    data['WEEK'] = data['DATE'].dt.to_period('W')
    return data

def post_predictive():
    '''
    Function to create posterior predictive intervals using the Binomial method
    '''
    return None
    
def logit_probs(parameter_samples, event_counts):
    '''
    Logistic probability finding function
    '''
    if 'gamma_' not in list(parameter_samples):
        alpha_list = parameter_samples['alpha'].values
        beta_list = parameter_samples['beta_'].values
        results = np.zeros((len(alpha_list), len(event_counts)))
        for vert, (alpha, beta) in enumerate(zip(alpha_list, beta_list)):
            for idx, count in enumerate(event_counts):
                results[vert, idx] = single_logit(count, [alpha, beta])
        return results
    else:
        alpha_list = parameter_samples['alpha'].values
        beta_list = parameter_samples['beta_'].values
        gamma_list = parameter_samples['gamma_'].values
        prev_zero, prev_one = np.zeros((len(alpha_list), len(event_counts))), np.zeros((len(alpha_list), len(event_counts)))
        for vert, (alpha, beta, gamma) in enumerate(zip(alpha_list, beta_list, gamma_list)):
            for idx, count in enumerate(event_counts):
                prev_zero[vert, idx] = general_logit([count, 0], [alpha, beta, gamma])
                prev_one[vert, idx] = general_logit([count, 1], [alpha, beta, gamma])
        return prev_zero, prev_one
    
def posterior_probabilty(sampled_parameters, observations):
    '''
    Function to find posterior probabilities given observations and sampled parameters
    '''
    return None

def box_violin_plot(observations, figsize=(10,10), box=True, violin=True, color=None, fig=None, ax=None):
    positions = range(observations.shape[1])
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)
    if violin:
        ax.violinplot(observations, widths=0.9, showmeans=True, positions=positions)
    if box:
        ax.boxplot(observations, widths=0.5, positions=positions)
    return fig, ax

def posterior_probability_plot(model, parameter_samples, event_range=5,
                               box=True, violin=True, plot_obs=False,
                               data=None, filter_dict=None, figsize=(10,10), color=None, alpha=0.5,
                               fig=None, ax=None):
    if not fig or not ax:
        fig, ax = plt.subplots(figsize=figsize)
    if model == 1:
        event_counts = list(range(1, event_range + 1))
        probabilties = logit_probs(parameter_samples, event_counts)
        positions = range(event_range)
        if violin:
            if color:
                parts = ax.violinplot(probabilties, widths=0.9, showmeans=True, positions=positions)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(alpha)
            else:
                ax.violinplot(probabilties, widths=0.9, showmeans=True, positions=positions)
        if box:
            ax.boxplot(probabilties, widths=0.5, positions=positions)
        if plot_obs:
            observed_proportions = observed_fatality_proportion(data, list(range(event_range)), filter_dict)
            for idx, observation in enumerate(observed_proportions):
                if observation:
                    ax.plot(idx, observation, 'ro')
        return fig, ax
    if model == 4:
        positions = range(1, parameter_samples.shape[1]+1)
        if violin:
            if color:
                parts = ax.violinplot(parameter_samples, widths=0.9, showmeans=True, positions=positions)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(alpha)
                for key in parts.keys():
                    if key != 'bodies':
                        parts[key].set_color(color)
            else:
                ax.violinplot(parameter_samples, widths=0.9, showmeans=True, positions=positions)
        if box:
            ax.boxplot(parameter_samples, widths=0.5, positions=positions)
    return fig, ax

def observed_fatality_proportion(data, event_counts, data_filter, prev_week=False):
    filtered_data = filter_data_by_dict(data, data_filter)
    agg_data = temporal_aggregation(filtered_data, 'WEEK')
    agg_data['PREV_WEEK'] = (agg_data['FATALITY_FLAG'].shift(1) > 0).astype(int)
    if prev_week:
        proportions = np.zeros((len(prev_week), len(event_counts)))
        for idx1, count in enumerate(event_counts):
            for idx2, value in enumerate(prev_week):
                total = len(agg_data.loc[agg_data['RAW_COUNTS'] == count].loc[agg_data['PREV_WEEK'] == value])
                fatal = len(agg_data.loc[agg_data['RAW_COUNTS'] == count].loc[agg_data['PREV_WEEK'] == value].loc[agg_data['FATALITY_FLAG'] == 1])
                if total != 0:
                    proportions[idx2, idx1] = fatal/total
                else:
                    proportions[idx2, idx1] = None
    else:
        proportions = np.zeros(len(event_counts))
        for idx, count in enumerate(event_counts):
            total = len(agg_data.loc[agg_data['RAW_COUNTS'] == count])
            fatal = len(agg_data.loc[agg_data['RAW_COUNTS'] == count].loc[agg_data['FATALITY_FLAG'] == 1])
            if total != 0:
                proportions[idx] = fatal/total
            else:
                proportions[idx] = None
    return proportions

def create_quantile_table(chains, quantiles=[0.25, 0.5, 0.75], parameter_names=None, mean=False):
    '''
    Table of quantiles creation function for parameters

    Arguments:
    chains - a data frame of sampled parameters
    quantiles - a list of quantiles to include in the data frame
    parameter_names - a list to optionally select only some of the columns of the data frame
    mean - a boolean toggle of whether to include the mean of the samples on the final row
    '''
    if not parameter_names:
        parameter_names = chains.columns
    quantile_table = pd.DataFrame(columns=['quantile'] + list(parameter_names))
    for quantile in quantiles:
        row = [f'{quantile * 100}%']
        for column in parameter_names:
            quantile_value = chains[column].quantile(quantile)
            row.append(quantile_value)
        quantile_table.loc[len(quantile_table)] = row
    if mean:
        row = ['mean']
        for column in parameter_names:
            mean_value = np.mean(chains[column])
            row.append(mean_value)
        quantile_table.loc[len(quantile_table)] = row
    return quantile_table

def burn_samples(samples, burn=500):
    '''
    Shorthand sample burning function - removes the first burn rows of the data frame
    '''
    return samples.iloc[burn:]

def simulate_data(model_type, temp_agg='WEEK', data=None, data_filter=None, parameters_mean=[], inner_event_mean=None, outer_event_mean=None,
                  num_obs=1000, random_events=True, ignore_zero=False, covariates_dict=None, max_bin=None):
    '''
    Data simulation function for models 1, 2, and 4-6. Functionality for model 5 is yet to be created, it is rather more complicated than I expected

    Arguments:
    model_type - model requiring data
    temp_agg - temporal aggregation used for simulation
    data - data frame for simulation. must provide either a data frame and filter or event means
    data_filter - dictionary filter for data frame
    parameters_mean - mean of parameters used for simulation of data
    inner_event_mean - mean of event counts in the district of choice/admin area in case of not model 5
    outer_event_mean - mean of event counts in the division of interest for model 5
    num_obs - number of simulated observations to generate
    random_events - toggle for whether to simulate events or use recorded event counts
    ignore_zero - toggle for ignoring weeks with zero events
    covariates_dict - dictionary of covariates for model 5
    max_bin - binning information for model 4


    '''
    if list(data):
        if random_events:
            if inner_event_mean:
                inner_events = np.random.poisson(inner_event_mean, size=num_obs)
            else:
                inner_event_mean = np.mean(model_observations(1, data=data, data_filter=data_filter, temp_agg=temp_agg, ignore_zero=ignore_zero)[1])
                inner_events = np.random.poisson(inner_event_mean, size=num_obs)
        else:
            inner_events = model_observations(1, data=data, data_filter=data_filter, temp_agg=temp_agg, ignore_zero=ignore_zero)[1]
    else:
        if inner_event_mean:
            inner_events = np.random.poisson(inner_event_mean, size=num_obs)
        else:
            raise ValueError('Must provide either a data frame or an inner event mean and a number of observations to generate')
    
    if model_type == 1:
        if len(parameters_mean) != 2:
            raise ValueError('Must provide two parameter means for Model 1')
        probabilities = [single_logit(event, parameters_mean) for event in inner_events]
        responses = np.random.binomial(1, probabilities)
        covariates = inner_events
        return responses, covariates
    
    if model_type == 2:
        if len(parameters_mean) != 3:
            raise ValueError('Must provide three parameter means for Model 2')
        fatalities = [np.random.binomial(1, single_logit(inner_events[0], parameters_mean[0:2]))]
        for idx in range(num_obs):
            prob = general_logit([inner_events[idx], fatalities[idx]], parameters_mean)
            fatalities.append(np.random.binomial(1, prob))
        responses = fatalities[1:]
        covariates = np.column_stack((inner_events[:-1], fatalities[:-1]))
        return responses, covariates
    
    if model_type == 4:
        if random_events:
            if inner_event_mean:
                inner_events = np.random.poisson(inner_event_mean, size=num_obs)
            else:
                inner_event_mean = np.mean(model_observations(1, data=data, data_filter=data_filter, temp_agg=temp_agg, ignore_zero=ignore_zero)[1])
                inner_events = np.random.poisson(inner_event_mean, size=num_obs)
            simulated_data = pd.DataFrame({'EVENT_COUNTS':inner_events})
            bins = [i for i in range(max_bin + 1)]
            bins.append(float('inf'))
            simulated_data = bin_data(simulated_data, bins=bins)
            event_bins = simulated_data['BINS'].values.astype(int)
        else:
            bins = [i for i in range(max_bin + 1)]
            bins.append(float('inf'))
            simulated_data = bin_data(data, bins=bins)
            event_bins = simulated_data['BINS'].values.astype(int)
        covariates = [count for count in event_bins if count > 0]
        responses = [np.random.binomial(1, parameters_mean[count-1]) for count in covariates]        
        return responses, covariates
    
    if model_type == 6:
        if not isinstance(parameters_mean, float):
            parameters_mean = parameters_mean[0]
        covariates = [count for count in inner_events if count > 0]
        responses = [np.random.binomial(1, 1-(1-parameters_mean)**count) for count in covariates]
        return responses, covariates