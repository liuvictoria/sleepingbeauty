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

import numpy as np
import pandas as pd

import bokeh
import bebi103

import tidy_data
import parameter_estimates

# +
##############
# get all dat good good
# DO NOT COMMENT OUT IF YOU WANT THE CODE TO RUN!!
# might fix in future so that we don't have all these rando constants twice
##############


#read the dataframe
df = tidy_data.tidy_dic()

#Labelled data
labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)']

#convert to ndarray. Will use in part b as well.
times_to_catastrophe = labeled_data.to_numpy()

#gamma distribution initial estimates / mle

#initial param estimates
alpha = 2
# find rate of arrivals
beta = alpha / np.mean(times_to_catastrophe)
init_params = [alpha, beta]

# test to check that we get reasonable MLEs
mle = parameter_estimates.mle_iid_gamma(times_to_catastrophe, init_params)

# conf ints
conf_intrvls = parameter_estimates.get_conf_intrvl_gamma(times_to_catastrophe, init_params, 1001)



#two arrival distribution intial estimates / mle
#set initial parameters for two arrival story
beta1 = 2 / np.sum(times_to_catastrophe) * len(times_to_catastrophe) - 0.00000001
init_params_two_events = [beta1, .00000001]

# first do beta_1 and delta_beta
mle_two_events = parameter_estimates.mle_iid_two_events(
    times_to_catastrophe, init_params_two_events, times_to_catastrophe
)

# test to check that we get reasonable MLEs
mle_two_events = np.array([
    mle_two_events[0],
    mle_two_events[0] + mle_two_events[1]
])

# conf ints
conf_intrvls_two_events = parameter_estimates.get_conf_intrvl_two_events(
    times_to_catastrophe, init_params_two_events, times_to_catastrophe, 1001
)
conf_intrvls_two_events = np.transpose(conf_intrvls_two_events)

# +
def plot_conf_intrvl_gamma():
    """
    plot 95% conf ints assuming gamma distribution
    Inputs:
        mle (ndarray of mle of each parameter)
        conf_int (ndarray of endpoints)
    """
    summary_alpha = [dict(label="alpha", estimate=mle[0], conf_int=conf_intrvls[0])]
    summary_beta = [dict(label="beta", estimate=mle[1], conf_int=conf_intrvls[1])]
    
    p1 = bebi103.viz.confints(
            summary_alpha,
            line_kwargs = {'color' : 'lightcoral'},
            marker_kwargs = {'color' : 'coral'}
    )
    p1.title.text = 'Confidence interval for alpha of gamma distribution'
    
    p2 = bebi103.viz.confints(
            summary_beta,
            line_kwargs = {'color' : 'plum'},
            marker_kwargs = {'color' : 'plum'}
    )
    p2.title.text = 'Confidence interval for beta of gamma distribution'
    
    return [p1, p2]

#make sure to run previously commented code to get mle and conf_ints
#uncomment code below to run
# plots = plot_conf_intrvl_gamma()
# bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=1))

# +
def plot_corner_gamma(df, init_param, size):
    """
    plot corner plot for alpha and beta for gamma distribution
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        95% conf int
    """
    #Labelled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)']

    #convert to ndarray. Will use in part b as well.
    times_to_catastrophe = labeled_data.to_numpy()
    
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    parameter_estimates.mle_iid_gamma, #func with scipy.optimize
                    parameter_estimates.gen_gamma, #generating function, after parameter estimates
                    times_to_catastrophe,
                    mle_args=(init_param,), #additional args to pass into mle_iid_gamma
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1
    )
    
    df_res = pd.DataFrame(data=mle_bs_reps, columns=["alpha*", "beta*"])
    
    # Eliminate possible erroneous solves (delete points outside of 99.9 percentile)
    inds = ((df_res < df_res.quantile(0.995)) & (df_res > df_res.quantile(0.015))).all(
        axis=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = bebi103.viz.corner(
            samples=df_res.loc[inds, :],
            parameters=["alpha*", "beta*"],
            show_contours=True,
            levels=[0.95],
            xtick_label_orientation = np.pi / 3,
            contour_color = 'indianred',
            single_var_color = 'rosybrown',
        )

    return p

# # uncomment to run
# p = plot_corner_gamma(df, init_params, 1001)
# bokeh.io.show(p)

# +
def plot_conf_intrvl_two_events():
    """
    plot 95% conf ints assuming gamma distribution
    Inputs:
        mle (ndarray of mle of each parameter)
        conf_int (ndarray of endpoints)
    """
    summary = [
        dict(label="beta 1", estimate=mle_two_events[0], conf_int=conf_intrvls_two_events[0]),
        dict(label="beta 2", estimate=mle_two_events[1], conf_int=conf_intrvls_two_events[1]),
    ]

    p1 = bebi103.viz.confints(
        summary,
        line_kwargs = {'color' : 'lightcoral'},
        marker_kwargs = {'color' : 'coral'}
    )
    
    p1.title.text = 'Confidence intervals for beta1 and beta2 of two event story'
    return p1

# #make sure to run previously commented code to get mle and conf_ints
# #uncomment code below to run
# plot = plot_conf_intrvl_two_events()
# bokeh.io.show(plot)

# +
def corner_plot_two_events(array, init_param, size):
    """
    plot corner plot for alpha and beta for gamma distribution
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        95% conf int
    """
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    parameter_estimates.mle_iid_two_events, 
                    parameter_estimates.gen_two_events,
                    array,
                    mle_args=(init_param, array),
                    gen_args = (array, ),
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1
    )
    
    df_res = pd.DataFrame(data=mle_bs_reps, columns=["beta_1*", "delta_beta*"])
    
    # Eliminate possible erroneous solves (delete points outside of 99.9 percentile)
    inds = ((df_res < df_res.quantile(0.995)) & (df_res > df_res.quantile(0.015))).all(
        axis=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = bebi103.viz.corner(
            samples=df_res.loc[inds, :],
            parameters=["beta_1*", "delta_beta*"],
            show_contours=True,
            levels=[0.95],
            xtick_label_orientation = np.pi / 3,
            contour_color = 'indianred',
            single_var_color = 'rosybrown',
        )
    return p

#uncomment to run
# p = corner_plot_two_events(times_to_catastrophe, init_params_two_events, 1001)
# bokeh.io.show(p)

# +
def plot_corner_two_events(array, init_param, size):
    """
    plot corner plot for alpha and beta for gamma distribution
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        95% conf int
    """
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    parameter_estimates.mle_iid_two_events, 
                    parameter_estimates.gen_two_events,
                    array,
                    mle_args=(init_param, array),
                    gen_args = (array, ),
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1
    )
    mle_bs_reps = np.transpose(mle_bs_reps)
    mle_bs_reps_betas = np.transpose(np.array([
        mle_bs_reps[0],
        mle_bs_reps[0] + mle_bs_reps[1]
    ]))
    
    df_res = pd.DataFrame(data=mle_bs_reps_betas, columns=["beta_1*", "beta_2*"])
    
    # Eliminate possible erroneous solves (delete points outside of 99.9 percentile)
    inds = ((df_res < df_res.quantile(0.995)) & (df_res > df_res.quantile(0.015))).all(
        axis=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = bebi103.viz.corner(
            samples=df_res.loc[inds, :],
            parameters=["beta_1*", "beta_2*"],
            show_contours=True,
            levels=[0.95],
            xtick_label_orientation = np.pi / 3,
            contour_color = 'indianred',
            single_var_color = 'rosybrown',
        )

    return p

# # #uncomment to run
# p = plot_corner_two_events(times_to_catastrophe, init_params_two_events, 1001)
# bokeh.io.show(p)

# +
def main():
    ##############
    # get all dat good good
    ##############

    #read the dataframe
    df = tidy_data.tidy_dic()

    #Labelled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)']

    #convert to ndarray. Will use in part b as well.
    times_to_catastrophe = labeled_data.to_numpy()

    #gamma distribution initial estimates / mle

    #initial param estimates
    alpha = 2
    # find rate of arrivals
    beta = alpha / np.mean(times_to_catastrophe)
    init_params = [alpha, beta]

    # test to check that we get reasonable MLEs
    mle = parameter_estimates.mle_iid_gamma(times_to_catastrophe, init_params)

    # conf ints
    conf_intrvls = parameter_estimates.get_conf_intrvl_gamma(times_to_catastrophe, init_params, 1001)



    #two arrival distribution intial estimates / mle
    #set initial parameters for two arrival story
    beta1 = 2 / np.sum(times_to_catastrophe) * len(times_to_catastrophe) - 0.00000001
    init_params_two_events = [beta1, .00000001]

    # first do beta_1 and delta_beta
    mle_two_events = parameter_estimates.mle_iid_two_events(
        times_to_catastrophe, init_params_two_events, times_to_catastrophe
    )

    # test to check that we get reasonable MLEs
    mle_two_events = np.array([
        mle_two_events[0],
        mle_two_events[0] + mle_two_events[1]
    ])

    # conf ints
    conf_intrvls_two_events = parameter_estimates.get_conf_intrvl_two_events(
        times_to_catastrophe, init_params_two_events, times_to_catastrophe, 1001
    )
    conf_intrvls_two_events = np.transpose(conf_intrvls_two_events)
    
    
    ############
    # plots
    ############
    
    plots = plot_conf_intrvl_gamma()
    bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=1))
    
    bokeh.io.save(
        plots,
        filename="viz_parameter_estimates_Fig3a.html",
        title="Parameter estimates for both distributions",
    )
    
    
    
    p1 = plot_corner_gamma(df, init_params, 1001)
    bokeh.io.show(p1)
    
    bokeh.io.save(
        p1,
        filename="viz_parameter_estimates_Fig3b.html",
        title="Parameter estimates for both distributions",
    )
    
    
    
    plot2 = plot_conf_intrvl_two_events()
    bokeh.io.show(plot2)
    
    bokeh.io.save(
        plot2,
        filename="viz_parameter_estimates_Fig3c.html",
        title="Parameter estimates for both distributions",
    )
    
    p3 = corner_plot_two_events(times_to_catastrophe, init_params_two_events, 1001)
    bokeh.io.show(p3)
    
    bokeh.io.save(
        p3,
        filename="viz_parameter_estimates_Fig3d.html",
        title="Parameter estimates for both distributions",
    )
    
    p4 = plot_corner_two_events(times_to_catastrophe, init_params_two_events, 1001)
    bokeh.io.show(p4)
    
    bokeh.io.save(
        p4,
        filename="viz_parameter_estimates_Fig3e.html",
        title="Parameter estimates for both distributions",
    )
    
    all_plots = [*plots, plot2,]
    bokeh.io.save(
        all_plots,
        filename="viz_parameter_estimates_Fig3.html",
        title="Parameter estimates for both distributions",
    )
    
    return True

if __name__ == '__main__' : main()

# +
#!jupytext --to python viz_parameter_estimates.ipynb
