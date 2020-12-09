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

import datashader

import bokeh.io
bokeh.io.output_notebook()

import holoviews as hv
import holoviews.operation.datashader
hv.extension('bokeh')
bebi103.hv.set_defaults()

import tidy_data
import parameter_estimates

from parameter_estimates import log_like_iid_gamma
from parameter_estimates import log_like_iid_two_events

from parameter_estimates import mle_iid_gamma
from parameter_estimates import mle_iid_two_events

from parameter_estimates import gen_gamma
from parameter_estimates import get_conf_intrvl_gamma

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

def _get_init_params_gamma():
    # for gamma distribution, each estimate refers to the estimate for that particular concentration
    alphas = np.ones(5,) * 2

    # find rate of arrivals
    # catastrophe_times is ndarray of catastrophe times for different concentrations
    betas = alphas / np.array(
        [catastrophe_time.mean() for catastrophe_time in catastrophe_times]
    )

    #transpose to get each array within the array as [alpha, beta]
    init_params = np.transpose([alphas, betas])
    

    return init_params

def gen_bs_alpha_beta(array, init_param, size):
    """
    draw bootstrap samples for alpha and beta of a gamma distribution
    using bebi103.bootstrap.draw_bs_reps_mle
    Only one concentration at a time
    Inputs:
        array : observed data
        init_param : tuple of initial estimates
        size : bootstrap rep size
    Outputs:
        mle_bs_reps
    """
    mle_bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
                    mle_iid_gamma, #func with scipy.optimize
                    gen_gamma, #generating function, after parameter estimates
                    array,
                    mle_args=(init_param,), #additional args to pass into mle_iid_gamma
                    size = size, 
                    progress_bar = False,
                    n_jobs = 1)
    return mle_bs_reps

# +
def _create_df_reps_mles():
    
    #get initial params
    init_param = _get_init_params_gamma()
    mles_bs_reps = []
    for i, concentration in enumerate(concentrations):
        #bs reps for beta and alpha
        mle_bs_reps = gen_bs_alpha_beta(
            catastrophe_times[i], 
            init_param[i],
            5000,
        )
        
        #manipulate for dataframe addition later
        #change to integer to add to ndarray
        concentration = int(concentration.split(' ')[0])
        mle_bs_reps = np.insert(mle_bs_reps, [0], concentration, axis = 1)
        mles_bs_reps.append(mle_bs_reps)
    
    #reshape by flattening the outermost layer
    mles_bs_reps = np.array([mles_bs_reps])
    mles_bs_reps = mles_bs_reps.reshape(-1, mles_bs_reps.shape[-1])
    
    #make dataframe
    df_reps = pd.DataFrame(
        data = mles_bs_reps,
        columns=['concentration', 'alpha', 'beta (1/s)']
    )
    return df_reps

# df_reps_mles = _create_df_reps_mles()
# -

def _create_df_mles_conf_ints():
    """
    dataframe for plotting hv alpha vs beta
    """
    #get initial params
    init_param = _get_init_params_gamma()
    df_rows = []
    for i, concentration in enumerate(concentrations):
        #manipulate for dataframe addition later
        #change to integer to add to ndarray
#         concentration = int(concentration.split(' ')[0])
        df_row_alpha, df_row_beta = [concentration, 'alpha',], [concentration, 'beta (1/s)',]
        
        #bs reps for beta and alpha
        conf_int = get_conf_intrvl_gamma(
            catastrophe_times[i], 
            init_param[i], 
            5000
        )
        alpha_low = conf_int[0][0]
        alpha_high = conf_int[0][1]
        df_row_alpha.append(alpha_low)
        df_row_alpha.append(alpha_high)

        beta_low = conf_int[1][0]
        beta_high = conf_int[1][1]
        df_row_beta.append(beta_low)
        df_row_beta.append(beta_high)
        
        
        #mle calculation 
        mle = mle_iid_gamma(catastrophe_times[i], init_param[i])
        df_row_alpha.append(mle[0])
        df_row_beta.append(mle[1])
        
        #append to master list
        df_rows.append(df_row_alpha)
        df_rows.append(df_row_beta)

    #make dataframe
    df_reps = pd.DataFrame(
        data = df_rows,
        columns=['concentration', 'parameter', 'conf_start', 'conf_end', 'mle']
    )
    return df_reps
# df_mle = _create_df_mles_conf_ints()
# df_mle

def ecdfs_alpha():
    """
    ecdf for alpha for different concentrations
    Output:
        bokeh figure
    """
    
    df_reps_mle = _create_df_reps_mles()
    
    # plot ecdfs using iqplot
    p = bokeh.plotting.figure(
        title = 'Alpha ECDF for different concentrations',
        width = 500,
        height = 400,
        x_axis_label = 'Alpha value',
        y_axis_label = 'ECDF',
        tooltips=[
        ("alpha value", "@{alpha}"),
    ],
    )
    
    iqplot.ecdf(
        data = df_reps_mle,
        q = 'alpha',
        cats = 'concentration',
        conf_int = True,
        palette = bokeh.palettes.Viridis5,
        p = p,
        conf_int_kwargs = {"fill_alpha": 0.35,}
        )
    p.legend.title = 'concen. (uM)'
    return p
# p = ecdfs_alpha()
# bokeh.io.show(p)

def ecdfs_beta():
    """
    ecdf for beta for different concentrations
    Output:
        bokeh figure
    """
    
    df_reps_mle = _create_df_reps_mles()
    
    # plot ecdfs using iqplot
    p = bokeh.plotting.figure(
        title = 'Beta ECDF for different concentrations',
        width = 500,
        height = 400,
        x_axis_label = 'Beta value (1/s)',
        y_axis_label = 'ECDF',
        tooltips=[
        ("beta value", "@{beta (1/s)}"),
    ],
    )
    
    iqplot.ecdf(
        data = df_reps_mle,
        q = 'beta (1/s)',
        cats = 'concentration',
        conf_int = True,
        palette = bokeh.palettes.Viridis5,
        p = p,
        conf_int_kwargs = {"fill_alpha": 0.35,}
        )
    p.legend.title = 'concen. (uM)'
    return p
# p = ecdfs_beta()
# bokeh.io.show(p)

# +
##########################################################
##########################################################
# Alpha / beta conf regions for different concentrations
#    Uses datashader and holoviews segments to plot alpha / beta conf regions
#    WARNING: TAKES A LONG TIME TO RUN. THINK A FEW MINUTES
##########################################################
##########################################################

df_reps_mles = _create_df_reps_mles()
df_mle = _create_df_mles_conf_ints()


colors = ['crimson', 'orange', 'greenyellow', 'forestgreen', 'blue']

# generate the base Points figure
points = hv.Points(
    data=df_reps_mles,
    kdims=['alpha', 'beta (1/s)'],
    vdims='concentration',
).groupby(
    'concentration'
).overlay(
)

# use datashader so we're not plotting tons of points
plot = hv.operation.datashader.dynspread(
    hv.operation.datashader.datashade(
        points,
        aggregator=datashader.by('concentration', datashader.count()),
        color_key = colors,
    )
)

# make segments to show range of D
D_segments = hv.NdOverlay(
    {
        concentration : hv.Segments(
            (
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'alpha'),'conf_start'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'beta (1/s)'),'mle'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'alpha'),'conf_end'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'beta (1/s)'),'mle'],
            ),
        ).opts(
            color = color,
            line_width = 2 
        )
        for concentration, color in zip(df_mle['concentration'].unique(), colors)
    }
).opts(
    title="Alpha and Beta Confidence Regions for Different Concentrations"
)

# make segments to show range of k_off
k_off_segments = hv.NdOverlay(
    {
        concentration: hv.Segments(
            (
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'alpha'),'mle'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'beta (1/s)'),'conf_start'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'alpha'),'mle'],
                df_mle.loc[(df_mle['concentration'] == concentration) & (df_mle['parameter'] == 'beta (1/s)'),'conf_end'],
            ),
        ).opts(
                color = color,
                line_width = 2
        )
        for concentration, color in zip(df_mle['concentration'].unique(), colors)
    }
)

my_panel = (plot * D_segments * k_off_segments).opts(frame_width=500, frame_height=400, show_grid=False)

# -

hv.save(my_panel, 'viz_concentration_effects_Fig6c.html', fmt = 'html',)

# +
def main():
    """
    warning takes a long time (minutes)
    """
    p1 = ecdfs_alpha()
    bokeh.io.show(p1)
    
    bokeh.io.save(
        p1,
        filename="viz_concentration_effects_Fig6a.html",
        title="Concentration influence on arrivals",
    )
    
    p2 = ecdfs_beta()
    bokeh.io.show(p2)
    
    bokeh.io.save(
        p2,
        filename="viz_concentration_effects_Fig6b.html",
        title="Concentration influence on arrivals",
    )
    
    all_plots = [p1, p2]
    
    bokeh.io.save(
        all_plots,
        filename="viz_concentration_effects_Fig6.html",
        title="Concentration influence on arrivals",
    )
    return True

if __name__ == '__main__': main()

# +
#!jupytext --to python viz_concentration_effects.ipynb
