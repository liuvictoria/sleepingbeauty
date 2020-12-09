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

import iqplot

import bebi103

import bokeh.io
bokeh.io.output_notebook()
import holoviews as hv
hv.extension('bokeh')

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
def plot_exploratory_ecdfs():
    """
    ecdfs of catastrophe times,
    colored by concentration
    
    Output:
        bokeh figure object
    """
    
    p = iqplot.ecdf(
        data = df,
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
    p.xaxis.axis_label = 'catastrophe times (s)'
    p.title.text = 'ECDF of catastrophe times by concentration'
    return p


# #uncomment to show
# p = plot_exploratory_ecdfs()
# bokeh.io.show(p)

# +
def plot_exploratory_strips():
    """
    strip box plots of catastrophe times,
    colored by concentration
    
    Output:
        bokeh figure object
    """   
    
    p2 = iqplot.stripbox(
        data = df,
        q = 'catastrophe time',
        cats = 'concentration',
        marker_kwargs=dict(alpha=0.3),
        jitter = True,
        show_legend = True,
        palette=bokeh.palettes.Magma7[1:-1][::-1],
        tooltips=[
            ('concentration', '@{concentration}'),
            ('catastrophe time', '@{catastrophe time}')
        ],
        y_axis_label = 'concentration',
    )
    p2.xaxis.axis_label = 'catastrophe times (s)'
    p2.title.text = 'Strip plot of catastrophe times by concentration'
    return p2

# #uncomment to show
# p2 = plot_exploratory_strips()
# bokeh.io.show(p2)

# +
def plot_exploratory_histograms():
    """
    histograms of catastrophe times,
    colored by concentration
    
    Output:
        bokeh figure object
    """

    p3 = iqplot.histogram(
        data = df,
        q = 'catastrophe time',
        cats = 'concentration',
        show_legend = True,
        palette=bokeh.palettes.Magma7[1:-1][::-1],
#         frame_width = 525,
    )
    p3.xaxis.axis_label = 'catastrophe times (s)'
    p3.title.text = 'Histogram of catastrophe times by concentration'
    return p3

# p3 = plot_exploratory_histograms()

# bokeh.io.show(p3)
# -

def main():
    p = plot_exploratory_ecdfs()
    bokeh.io.show(p)
    
    bokeh.io.save(
        p,
        filename="viz_explore_concentration_datset_Fig4a.html",
        title="Exploratory analysis ",
    )
    
    
    p2 = plot_exploratory_strips()
    bokeh.io.show(p2)
    
    bokeh.io.save(
        p2,
        filename="viz_explore_concentration_datset_Fig4b.html",
        title="Parameter estimates for both distributions",
    )
    
    
    p3 = plot_exploratory_histograms()
    bokeh.io.show(p3)
    
    bokeh.io.save(
        p3,
        filename="viz_explore_concentration_datset_Fig4c.html",
        title="Parameter estimates for both distributions",
    )
    
    all_plots = [p, p2, p3]
    
    bokeh.io.save(
        all_plots,
        filename="viz_explore_concentration_datset_Fig4.html",
        title="Parameter estimates for both distributions",
    )
    
    
    return True
if __name__ == '__main__': main()
    

# +
#!jupytext --to python viz_explore_concentration_dataset.ipynb
# -


