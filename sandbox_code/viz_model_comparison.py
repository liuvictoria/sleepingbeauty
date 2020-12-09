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
import warnings

import pandas as pd
import numpy as np

import scipy.stats
import iqplot

import bebi103

import bokeh.io
bokeh.io.output_notebook()
import holoviews as hv
hv.extension('bokeh')

import tidy_data
import parameter_estimates

from parameter_estimates import log_like_iid_gamma
from parameter_estimates import log_like_iid_two_events

from parameter_estimates import mle_iid_gamma
from parameter_estimates import mle_iid_two_events

data_path = "../data/"
rg = np.random.default_rng()

# +
##########
# Don't comment out!!1!!
##########

df = tidy_data.tidy_concentrations()

#made ndarrays of various concentrations
concentrations = df.concentration.unique()

#make ndarrays of catastrophe times for different concentrations
#catastrophe_times[0] is for the lowest concentration, while cat_times[4] is for the highest concentration
catastrophe_times = np.array([
  df.loc[df['concentration'] == concent, 'catastrophe time'] for concent in df.concentration.unique()
])
# -

def gamma_mles(printme = False):
    """
    Sanity check to see what the MLEs are.
    Are they reasonable?
    
    Inputs:
        printme : True if want to print out nicely
    """
    # for gamma distribution, each estimate refers to the estimate for that particular concentration
    alphas = np.ones(5,) * 2

    # find rate of arrivals
    # catastrophe_times is ndarray of catastrophe times for different concentrations
    betas = alphas / np.array(
        [catastrophe_time.mean() for catastrophe_time in catastrophe_times]
    )

    #transpose to get each array within the array as [alpha, beta]
    init_params = np.transpose([alphas, betas])
    
    
    # test to check that we get reasonable MLEs
    mles = []
    for i in range(5):
        mle = mle_iid_gamma(catastrophe_times[i], init_params[i])
        mles.append(mle)

    mles = np.transpose(mles)
    alphas, betas = mles
    if printme:
        for i in range(5):
            print(f'concentration: {concentrations[i]}')
            print("""gamma distribution estimates: \nalpha: {0:.6f} \nbeta: {1:.6f} \n""".format(
                alphas[i], betas[i]
                ))
    return mles

def two_events_mles(printme = False):
    """
    Sanity check to see what the MLEs are.
    Are they reasonable?
    
    Inputs:
        printme : True if want to print out nicely
    """
    
    #mle is where beta1 is appx equal to beta2
    beta1s = np.array([
        2 / np.sum(catastrophe_time) * len(catastrophe_time)
        for catastrophe_time in catastrophe_times
        ])
    #so deltabeta is very small
    delta_betas = np.ones(5,) * .00000001

    #transpose to get [beta1, deltabeta] form, to pass into functions
    init_params_two_events = np.transpose([beta1s, delta_betas])

    # test to check that we get reasonable MLEs
    mles_two_events = []
    for i in range(5):
        mle_two_events = mle_iid_two_events(
            catastrophe_times[i], 
            init_params_two_events[i], 
            catastrophe_times[i],
        )
        mles_two_events.append(mle_two_events)

    mles_two_events = np.transpose(mles_two_events)

    #get beta1s and beta2s from beta1 and deltabeta mle estimates
    beta1s = mles_two_events[0]
    beta2s = mles_two_events[0] + mles_two_events[1]
    mle_two_events = np.array([beta1s, beta2s])
    
    if printme:
        #print for sanity check
        for i in range(5):
            print(f'concentration: {concentrations[i]}')
            print("""two arrival parameter estimates: \nbeta1: {0:.10f} \nbeta2: {1:.10f} \n""".format(
                beta1s[i], beta2s[i]
                ))
            
    return mle_two_events

def gen_two_events_3D(params, data, size, rg):
    """
    generative func for two events
    Inputs:
        params: necessary for calling function
            beta_1, delta_beta = params
        data : observed data, only needed for size
        size : number of bootstrap samples
    Outputs:
        np.array(
            [np.array(rg.gamma(alpha, 1 / beta_1, size=data_size)) for _ in range(size)]
        )
    """  
    beta_1, _ = params
    alpha = 2
    data_size = len(data)
    return np.array(
        [np.array(rg.gamma(alpha, 1 / beta_1, size=data_size)) for _ in range(size)]
    )

def gen_gamma_3D(params, data, size, rg):
    """
    generative func for gamma
    Inputs:
        params: necessary for calling function
            alpha, beta = params
        data : observed data, only needed for size
        size : number of bootstrap samples
    Outputs:
        np.array(
            [rg.gamma(alpha, 1 / beta, size=data_size) for _ in range(size)]
        )
    Notes:
        size should equal size of data for qq plots
    """
    alpha, beta = params
    data_size = len(data)
    return np.array(
        [rg.gamma(alpha, 1 / beta, size=data_size) for _ in range(size)]
    )

def create_df_mle():
    """
    df of mles of alpha, beta, beta1, beta2
    for 7, 9, 10, 12, 14 uM concentrations
    """
    mles = gamma_mles()
    mles_two_events = two_events_mles()
    mle_estimates = np.concatenate((mles, mles_two_events))
    df_mle = pd.DataFrame(
        mle_estimates, 
        index=['alpha', 'beta', 'beta1', 'beta2',], 
        columns = concentrations
        )
    return df_mle
# df_mle = create_df_mle()
# df_mle

def create_df_aic(df_mle):
    """
    df of mles of alpha, beta, beta1, beta2;
    log likelihood and AIC
    for 7, 9, 10, 12, 14 uM concentrations
    
    """
    
    
    #AIC get log likelihood
    #for gamma first
    rg = np.random.default_rng()
    for concentration in concentrations:
        params = (
            df_mle.loc['alpha', concentration],
            df_mle.loc['beta', concentration],
        )
        n = df.loc[df['concentration'] == concentration, 'catastrophe time'].values
        log_likelihood_gamma = log_like_iid_gamma(params, n)
        df_mle.loc['log_like_gamma', concentration] = log_likelihood_gamma

    #now for two events
    for i, concentration in enumerate(concentrations):
        params = (
            df_mle.loc['beta1', concentration],
            df_mle.loc['beta2', concentration],
        )
        n = df.loc[df['concentration'] == concentration, 'catastrophe time'].values
        log_likelihood_two_events = log_like_iid_two_events(params, n, n)
        df_mle.loc['log_like_two', concentration] = log_likelihood_two_events


    #now actually get AIC
    for concentration in concentrations:
        df_mle.loc['AIC_gamma', concentration] = -2 * (df_mle.loc['log_like_gamma', concentration] - 2)
        df_mle.loc['AIC_two', concentration] = -2 * (df_mle.loc['log_like_two', concentration] - 2)
        
    
    #get AIC weights
    for concentration in concentrations:
        AIC_max = max(df_mle.loc[['AIC_gamma', 'AIC_two'], concentration])
        numerator = np.exp(-(df_mle.loc['AIC_gamma', concentration] - AIC_max)/2)
        denominator = numerator + np.exp(-(df_mle.loc['AIC_two', concentration] - AIC_max)/2)
        df_mle.loc['w_gamma', concentration] = numerator / denominator
        df_mle.loc['w_two', concentration] = 1 - numerator / denominator
        
    return df_mle
# # Take a look
df_mle = create_df_mle()
df_mle = create_df_aic(df_mle)
df_mle

# +
def gen_gamma_predictives(df_mle):
    """
    data samples out of generative gamma distribution;
    parametrized by mles 
    """

    gamma_predictives = []
    for concentration in concentrations:
        params = (
            df_mle.loc['alpha', concentration],
            df_mle.loc['beta', concentration],
        )
        data = df.loc[df['concentration'] == concentration, 'catastrophe time'].values
        gamma_predictives.append(gen_gamma_3D(params, data, 1000, rg))
    return gamma_predictives

    
    
def gen_two_events_predictives(df_mle):
    """
    data samples out of generative two story distribution;
    parametrized by mles
    """  
    
    #data samples out of generative two story distribution
    two_events_predictives = []
    for concentration in concentrations:
        params = (
            df_mle.loc['beta1', concentration],
            df_mle.loc['beta2', concentration],
        )
        data = df.loc[df['concentration'] == concentration, 'catastrophe time'].values
        two_events_predictives.append(gen_two_events_3D(params, data, 1000, rg))
    return two_events_predictives

df_mle = create_df_mle()

# gamma_predictives = gen_gamma_predictives(df_mle)
# two_events_predictives = gen_two_events_predictives(df_mle)

# +
def plot_predictive_ecdfs():
    """
    predictive ecdfs for all tubulin concentrations, for catastrophe time
    for both gamma and two event stories
    """
    df_mle = create_df_mle()
    gamma_predictives = gen_gamma_predictives(df_mle)
    two_events_predictives = gen_two_events_predictives(df_mle)
    
    #generate predictive ecdfs plots

    #first for gamma distribution
    gamma_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'gamma model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label="catastrophe times (s)",
            y_axis_label="ECDF",
        )
        bebi103.viz.predictive_ecdf(
            data=catastrophe_times[i],
            samples=gamma_predictives[i],
            p = p,
            color = 'gray',
            data_color = 'lightcoral',
        )
        gamma_plots.append(p)

    #now for two event story
    two_event_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'two event model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label="catastrophe times (s)",
            y_axis_label="ECDF",
        )
        bebi103.viz.predictive_ecdf(
            data=catastrophe_times[i],
            samples=two_events_predictives[i],
            p = p,
            color = 'gray',
            data_color = 'lightcoral',
        )

        two_event_plots.append(p)
    plots = []
    plots = list((np.transpose([gamma_plots] + [two_event_plots]).flatten()))
    return plots

# plots = plot_predictive_ecdfs()
# bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))
# -

def plot_qq():
    """
    plot qq plots for all concentrations and both models
    """
    df_mle = create_df_mle()
    gamma_predictives = gen_gamma_predictives(df_mle)
    two_events_predictives = gen_two_events_predictives(df_mle)
    
    #generate qq plots

    #first for gamma distribution
    gamma_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'gamma model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label="catastrophe times (s)",
            y_axis_label="catastrophe times (s)",
        )
        bebi103.viz.qqplot(
            data=catastrophe_times[i],
            samples=gamma_predictives[i],
            p = p,
            line_kwargs = {'color':'pink',},
            diag_kwargs = {'color':'darkslategray',},
            patch_kwargs = {"fill_alpha": 0.3, 'color':'lightcoral',},
        )
        gamma_plots.append(p)

    #now for two event story
    two_event_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'two event model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label="catastrophe times (s)",
            y_axis_label="catastrophe times (s)",
        )
        bebi103.viz.qqplot(
            data=catastrophe_times[i],
            samples=two_events_predictives[i],
            p = p,
            line_kwargs = {'color':'pink',},
            diag_kwargs = {'color':'darkslategray',},
            patch_kwargs = {"fill_alpha": 0.3, 'color':'lightcoral',},
        )

        two_event_plots.append(p)
    plots = []
    plots = list((np.transpose([gamma_plots] + [two_event_plots]).flatten()))
    return plots
# plots = plot_qq()
# bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))

# +
def plot_predictive_ecdfs_diff():
    """
    plot predictive ecdfs for all concentrations and both models
    """
    df_mle = create_df_mle()
    gamma_predictives = gen_gamma_predictives(df_mle)
    two_events_predictives = gen_two_events_predictives(df_mle)
    
    #generate predictive ecdfs

    #first for gamma distribution
    gamma_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'gamma model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label = 'catastrophe times (s)'
        )
        bebi103.viz.predictive_ecdf(
            data=catastrophe_times[i],
            samples=gamma_predictives[i],
            diff = 'ecdf',
            discrete = 'True',
            p = p,
            color = 'gray',
            data_color = 'lightcoral',

        )
        gamma_plots.append(p)

    #now for two event story
    two_event_plots = []
    for i in range(5):
        p = bokeh.plotting.figure(
            title = f'two event model for {concentrations[i]}',
            width=400,
            height=300,
            x_axis_label="catastrophe times (s)",
        )
        bebi103.viz.predictive_ecdf(
            data=catastrophe_times[i],
            samples=two_events_predictives[i],
            diff = 'ecdf',
            discrete = True,
            p = p,
            color = 'gray',
            data_color = 'lightcoral',
        )

        two_event_plots.append(p)
    plots = []
    plots = list((np.transpose([gamma_plots] + [two_event_plots]).flatten()))
    return plots

# plots = plot_predictive_ecdfs_diff()
# bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))

# +
def main():
    plots = plot_predictive_ecdfs()
    bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=2))
    
    bokeh.io.save(
        plots,
        filename="viz_model_comparison_Fig5a.html",
        title="Graphic model comparison",
    )
    
    
    plots2 = plot_qq()
    bokeh.io.show(bokeh.layouts.gridplot(plots2, ncols=2))
    
    bokeh.io.save(
        plots2,
        filename="viz_model_comparison_Fig5b.html",
        title="Graphic model comparison",
    )
    
    
    plots3 = plot_predictive_ecdfs_diff()
    bokeh.io.show(bokeh.layouts.gridplot(plots3, ncols=2))
    
    bokeh.io.save(
        plots3,
        filename="viz_model_comparison_Fig5c.html",
        title="Graphic model comparison",
    )
    
    all_plots = [*plots, *plots2,  *plots3]
    bokeh.io.save(
        all_plots,
        filename="viz_model_comparison_Fig5.html",
        title="Graphic model comparison",
    )
    
    return True

if __name__ == '__main__': main()

# +
#!jupytext --to python viz_model_comparison.ipynb
# -


