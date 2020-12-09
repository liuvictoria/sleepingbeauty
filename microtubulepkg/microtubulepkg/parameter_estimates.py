# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os, sys, subprocess
import warnings

import pandas as pd
import numpy as np

import scipy.stats
import scipy.optimize

import numba
import bebi103
import tidy_data


data_path = "../data/"
rg = np.random.default_rng()
# -



def log_like_iid_gamma(params, n):
    """
    Calculates log likelihood for iid gamma measurements.
    
    Inputs: 
        params : tuple or list
            Must have 2 elements. In order, mean of alpha, beta
        n : ndarray of observed catastrophe times
    Outputs: 
        log-likelihood.
    """
    for param in params: 
        if param <= 0:
            return -np.inf
    alpha, beta = params
    
    return np.sum(scipy.stats.gamma.logpdf(n, alpha, loc=0, scale=1 / beta))


# +
def mle_iid_gamma(n, x0):
    """
    Calculates the MLE for a gamma distribution. 
    
    Inputs: 
        n: ndarray of observations / data that we are modeling with the parameters
        x0 : 1x2 ndarray
            Initial estimates for the optimization.
            Must have 2 elements. In order, alpha, beta.
    Returns: 
        res.x : list
            MLE for each parameter, in same order as x0. 
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=x0,
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

        
#uncomment to run
#initial param estimates
# alpha = 2
# # find rate of arrivals
# beta = alpha / np.mean(times_to_catastrophe)
# init_params = [alpha, beta]
# # test to check that we get reasonable MLEs
# mle = mle_iid_gamma(times_to_catastrophe, init_params)
# output
#array([2.40754885, 0.00546287])

# -

def gen_gamma(params, size, rg):
    """
    generating function for gamma distribution
    Inputs:
        params (ndarray) alpha, beta
        size : number of points to draw
        rg : random number generator (np.random.default_rng)
    Outputs:
        values from gamma distribution, parametrized by params
    """
    alpha, beta = params
    return rg.gamma(alpha, 1 / beta, size=size)

# +
def get_conf_intrvl_gamma(array, init_param, size):
    """
    draw conf ints for alpha and beta of a gamma distribution
    using bebi103.bootstrap.draw_bs_reps_mle
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        95% conf int
    """
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    mle_iid_gamma, #func with scipy.optimize
                    gen_gamma, #generating function, after parameter estimates
                    array,
                    mle_args=(init_param,), #additional args to pass into mle_iid_gamma
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1)

    
    return np.transpose(np.percentile(mle_bs_reps, [2.5, 97.5], axis=0))

#uncomment to run
# conf_intrvls = get_conf_intrvl_gamma(times_to_catastrophe, init_params, 1001)
# conf_intrvls
# #outputs
# alpha conf int: [1.99190, 2.90202] 
# beta conf int : [0.00454, 0.00676]
# -

def log_like_iid_two_events(params, n, original_data):
    """
    Calculates log likelihood for iid gamma measurements.
    
    Inputs: 
        params : tuple or list
            Must have 2 elements. In order, beta_1, delta_beta
        n : the data that we have
        y : n x 1 numpy array
            
    Outputs: 
        log-likelihood.
    """
    #beta1 and delta_beta can't be negative.
    #Return -inf since we want to maximize positive values only
    for param in params: 
        if param < 0:
            return -np.inf
    
    beta_1, delta_beta = params
    if beta_1 > 2 / np.mean(original_data):
        return -np.inf
    
    #if deltabeta approches 0, the model collapses to a gamma distribution
    # gamma distribution with alpha = 2 and beta = beta1 (or beta2)
    if np.isclose(delta_beta, 0):
        return np.sum(scipy.stats.gamma.logpdf(n, 2, loc=0, scale=1 / beta_1))
    
    
    #if not close, we'll return the actual log likelihood of the two-process story
    length = len(n)
    tmp1 = length * np.log(beta_1) + length * np.log(beta_1 + delta_beta) - length * np.log(delta_beta)
    tmp2 = -np.sum(beta_1 * n) + np.sum(np.log(1 - np.exp(-delta_beta * n)))
    return tmp1 + tmp2

def mle_iid_two_events(n, x0, original_data):
    """
    Calculates the MLE for a gamma distribution. 
    
    Inputs: 
        n  : ndarray of observations of data
        x0 : 1x2 ndarray
            Initial estimates for the optimization.
            Must have 2 elements. In order, beta_1, delta_beta
    Returns: 
        res.x : list
            MLE for each parameter, in same order as x0. 
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n, original_data: -log_like_iid_two_events(params, n, original_data),
            x0=x0,
            args=(n, original_data),
            method='Powell',
            tol=1e-8,
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


# +
# #uncomment to test out
# #set initial parameters for two arrival story
# beta1 = 2 / np.sum(times_to_catastrophe) * len(times_to_catastrophe) - 0.00000001
# init_params_two_events = [beta1, .00000001]

# # first do beta_1 and delta_beta
# mle_two_events = mle_iid_two_events(times_to_catastrophe, init_params_two_events, times_to_catastrophe)

# # test to check that we get reasonable MLEs
# mle_two_events = np.array([
#     mle_two_events[0],
#     mle_two_events[0] + mle_two_events[1]
# ])

# mle_two_events

# # # outputs
# # # beta1 mle: 0.0045381223
# # # delta beta mle: 0.0000000062

# # # outputs
# # # beta1 mle: 0.0045381223
# # # beta2 mle: 0.0045381285
# -

def gen_two_events(params, n, size, rg):
    """
    Inputs:
        params: ndarray, beta_1, delta_beta = params
        size : number of bootstrap samples
        
    """  
    beta_1, delta_beta = params
    beta_2 = beta_1 + delta_beta
    return rg.exponential(1/beta_1, size = size) + rg.exponential(1/beta_2, size = size)

# +
def get_conf_intrvl_two_events(array, init_param, original_data, size):
    """
    draw conf ints for beta1 and beta2 of two story distribution
    using bebi103.bootstrap.draw_bs_reps_mle
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        95% conf int
    """
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    mle_iid_two_events, 
                    gen_two_events,
                    array,
                    mle_args=(init_param, original_data,),
                    gen_args = (array, ),
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1)
    #work with individual arrays containing all estimates of a single parameter
    mle_bs_reps = np.transpose(mle_bs_reps)
    
    beta_1s = mle_bs_reps[0]
    beta_2s = beta_1s + mle_bs_reps[1]

    
    return np.percentile(np.array(
        [beta_1s, beta_2s]
        ), [2.5, 97.5], axis=1)

# #uncomment to run
# conf_intrvls_two_events = get_conf_intrvl_two_events(times_to_catastrophe, init_params_two_events, times_to_catastrophe, 1001)
# conf_intrvls_two_events = np.transpose(conf_intrvls_two_events)

# #output:
# # beta1 conf int: [0.003227, 0.004538] 
# # beta2 conf int : [0.004132, 0.008896]

# +
def main():
    df = tidy_data.tidy_dic()

    #Labelled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)']

    #convert to ndarray. Will use in part b as well.
    times_to_catastrophe = labeled_data.to_numpy()
    
    ###### gamma:
    alpha = 2
    # find rate of arrivals
    beta = alpha / np.mean(times_to_catastrophe)
    init_params = [alpha, beta]
    # test to check that we get reasonable MLEs
    mle = mle_iid_gamma(times_to_catastrophe, init_params)
    #confints
    conf_intrvls = get_conf_intrvl_gamma(times_to_catastrophe, init_params, 1001)
    
    
    ####### two events:
    #set initial parameters for two arrival story
    beta1 = 2 / np.sum(times_to_catastrophe) * len(times_to_catastrophe) - 0.00000001
    init_params_two_events = [beta1, .00000001]

    # first do beta_1 and delta_beta
    mle_two_events = mle_iid_two_events(times_to_catastrophe, init_params_two_events, times_to_catastrophe)

    # test to check that we get reasonable MLEs
    mle_two_events = np.array([
        mle_two_events[0],
        mle_two_events[0] + mle_two_events[1]
    ])
    #confints
    conf_intrvls_two_events = get_conf_intrvl_two_events(times_to_catastrophe, init_params_two_events, times_to_catastrophe, 1001)
    conf_intrvls_two_events = np.transpose(conf_intrvls_two_events)
    
    
    ######### print statements
    
    #gamma first
    print("""initial parameters for \nalpha: {0} \nbeta: {1:.6f}""".format(*init_params))
    
    print(
    """
    alpha conf int: [{0:.5f}, {1:.5f}] \n
    beta conf int : [{2:.5f}, {3:.5f}]
    """.format(
    *conf_intrvls.flatten()
    )
    )
    
    #two story
    print(
    """
    beta1 mle: {0:.10f}\n
    beta2 mle: {1:.10f}
    """.format(
    mle_two_events[0],
    mle_two_events[1],
    )
    )
    
    print(
        """
        beta1 conf int: [{0:.6f}, {1:.6f}] \n
        beta2 conf int : [{2:.6f}, {3:.6f}]
        """.format(
        *conf_intrvls_two_events.flatten()
        )
    )
    return True

if __name__ == '__main__': main()
# -

#!jupytext --to python parameter_estimates.ipynb

