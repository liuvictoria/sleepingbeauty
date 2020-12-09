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

import scipy.stats as st

import iqplot

import bebi103

import bokeh.io
bokeh.io.output_notebook()
import holoviews as hv
hv.extension('bokeh')
import panel as pn

import tidy_data

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

# +
def plot_overlaid_ecdfs(alpha, time, concentration):
    """
    ecdfs of catastrophe times,
    colored by concentration
    also includes gamma distribution overlaid
    Output:
        bokeh figure object
    """
    if concentration != 'all':
        sub_df = df.loc[df['concentration_int'] == concentration]
    else:
        sub_df = df
    
    #plot actual data
    p = iqplot.ecdf(
        data = sub_df,
        q = 'catastrophe time',
        cats = 'concentration',
        marker_kwargs=dict(line_width=0.3, alpha = 0.6),
        show_legend = True,
        palette=bokeh.palettes.Magma8[1:-2][::-1],
        tooltips=[
            ('concentration', '@{concentration}'),
            ('catastrophe time', '@{catastrophe time}')
        ],
    )
    p.xaxis.axis_label = "catastrophe times (s)"
    
    #get points to plot line
    x = np.linspace(0, 2000)
    y = st.gamma.cdf(x, alpha, scale=time)
    
    #overlay ecdf, can be scaled by widgets
    p.line(
        x = x,
        y = y,
        color = 'yellowgreen',
        width = 3
    )
    
    p.title.text = 'ECDF of catastrophe times by concentration'
    return p


# #uncomment to show
# p = plot_exploratory_ecdfs()
# bokeh.io.show(p)

# +
alpha_slider = pn.widgets.FloatSlider(
    name='alpha',
    start=1.8,
    end=4.2,
    step=0.1,
    value=2.4075
)

time_slider = pn.widgets.FloatSlider(
    name='average interarrival time (s)',
    start=75,
    end=215,
    step=10,
    value=1 / 0.005462
)

concentration_slider = pn.widgets.Select(
    name='concentration (uM)',
    options=[7, 9, 10, 12, 14, 'all',],
    value = 'all',
)
# -

@pn.depends(
    alpha=alpha_slider.param.value, 
    time=time_slider.param.value,
    concentration = concentration_slider.param.value
)
def plot_overlaid_ecdfs_pn(alpha, time, concentration):
    return plot_overlaid_ecdfs(alpha, time, concentration)

widgets = pn.Column(pn.Spacer(width=30), alpha_slider, time_slider, concentration_slider, width=500)
panel = pn.Column(plot_overlaid_ecdfs_pn, widgets)
panel.save('interactive', embed = True, max_opts = 40)

def main():
    p = plot_overlaid_ecdfs(2, 1 / .005, 'all')
    bokeh.io.show(p)

    
    
    return True
if __name__ == '__main__': main()


# +
#!jupytext --to python viz_dashboard.ipynb
