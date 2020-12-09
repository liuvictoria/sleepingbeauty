# -*- coding: utf-8 -*-
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

"""
code to generate Figure 2, which 
explores what different values of beta1 and beta2 affect
the catastrophe times the two-arrival story.
Also has a plot of brentq roots for MLE of beta1 / beta2
"""

# +
import os, sys, subprocess

import tidy_data

import pandas as pd
import numpy as np

import scipy.stats
import iqplot

import panel as pn

import bokeh.io
bokeh.io.output_notebook()
pn.extension()

data_path = "../data/"
# -

def _create_df_for_dashboard():
    """
    create a dataframe of various values of beta1 and beta2
    and the corresponding time to catastrophe.
    Modeled by a joint exponential distribution that is added together
    
    Inputs:
        None
    Outputs:
        df
    
    Notes: This is not user-facing, since we use it for plotting the dashboard
    that explores varying values of beta1 and beta2
    """
    # set possible values for beta1 and beta2
    # recall these are rates, so beta1s[2] corresponds to 1 per 100 seconds
    beta1s = [1,1/30,1/100,1/300,1/1000,1/3000]
    beta2s = [1,1/30,1/100,1/300,1/1000,1/3000]

    # make an empty ndarray to store our samples
    samples = np.ndarray((0,3))
    
    rg = np.random.default_rng()
    
    for i, beta1 in enumerate(beta1s):
        for j, beta2 in enumerate(beta2s):

            # draw times; numpy takes 1/beta for the scale
            t1 = rg.exponential(1/beta1, size=150)
            t2 = rg.exponential(1/beta2, size=150)

            catast_t = [(t1[k] + t2[k], beta1, beta2) for k in range(len(t1))]

            # store the samples
            samples = np.concatenate((samples, catast_t))

    # move samples into a dataframe
    df = pd.DataFrame(data=samples, columns=['time to catastrophe (s)','beta1','beta2'])
    return df

def _extract_sub_df(df, beta1, beta2):
    """
    Pulls data from df corresponding to the chosen beta1 and beta2.
    """
    
    inds = (
        np.isclose(df["beta1"], beta1)
        & np.isclose(df["beta2"], beta2)
    )
    
    return df.loc[inds, :]

def plot_ecdf(df, beta1, beta2):
    """
    Dashboarding.
    Generates the ECDF for the chosen beta1 and beta2.
    """
    
    sub_df = _extract_sub_df(df, beta1, beta2)
    
    return iqplot.ecdf(data=sub_df, q="time to catastrophe (s)")

# +
#     uncomment to use code
df = _create_df_for_dashboard()

beta1_slider = pn.widgets.DiscreteSlider(
    name='beta1',
    options=list(df["beta1"].unique()),
    value = 1.0
)

beta2_slider = pn.widgets.DiscreteSlider(
    name='beta2',
    options=list(df["beta2"].unique()),
    value = 1.0
)

# Make the ECDF depend on the selected values of beta1 and beta2

@pn.depends(
    beta1=beta1_slider.param.value, 
    beta2=beta2_slider.param.value
)
def plot_ecdf_pn(beta1, beta2):
    return plot_ecdf(df, beta1, beta2)
# -

def _draw_model(beta_1, beta_2, size=1):
    """
    draw out of two exponential distributions
    parametrized by beta_1 and beta_2
    """
    rg = np.random.default_rng()
    return rg.exponential(1/beta_1, size=size) + rg.exponential(1/beta_2, size=size)

# +
def plot_beta_ratios_ecdf():
    """
    different expected catastrophe times for different ratios of beta1/beta2;
    ratio range of [0.1, 0.3, 1, 3, 10]
    Inputs:
        None
    Outputs:
        Bokeh figure
    """
    n_samples = 150
    p = None

    p = bokeh.plotting.figure(
        frame_height=300,
        frame_width=450,
        x_axis_label="time to catastrophe × β₁",
        y_axis_label="ECDF",
    )

    beta_ratio = [0.1, 0.3, 1, 3, 10]

    catastrophe_times = np.concatenate(
        [_draw_model(1, br, size=n_samples) for br in beta_ratio]
    )
    beta_ratios = np.concatenate([[br] * n_samples for br in beta_ratio])
    df = pd.DataFrame(
        data={"β₂/β₁": beta_ratios, "time to catastrophe × β₁": catastrophe_times}
    )

    p = iqplot.ecdf(
        df,
        q="time to catastrophe × β₁",
        cats="β₂/β₁",
        palette=bokeh.palettes.Magma7[1:-1][::-1],
    )
    p.legend.title = "β₂/β₁"
    p.title.text = 'β₂/β₁ ratio effect on joint exponential distribution'
    return p

#to use, uncomment this
# p = plot_beta_ratios_ecdf()
# bokeh.io.show(p)
# -

def _dl_ddb(beta_1, times_to_catastrophe):
    """
    coding up d(ell)/db
    """
    t_bar = times_to_catastrophe.mean()
    n = len(times_to_catastrophe)
    delta_beta = beta_1 * (t_bar * beta_1 - 2) / (1 - t_bar * beta_1)
    tmp1 = n / (beta_1 + delta_beta)
    tmp2 = n / delta_beta
    tmp3 = np.sum(
        times_to_catastrophe * np.exp(-delta_beta * times_to_catastrophe) / (1 - np.exp(-delta_beta * times_to_catastrophe))
    )
    return tmp1 - tmp2 + tmp3

# +
def plot_brentq_roots():
    """
    plot the derivative of the log likelihood with respect to
    delta beta, as parametrized by beta_1, to see where
    the roots are. Plotted over values of beta_1 (0.003, 0.008)
    
    Inputs:
        None
    Outputs:
        bokeh figure
    """
    
    df = tidy_data.tidy_dic()
    
    #Labelled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)']

    #convert to ndarray
    times_to_catastrophe = labeled_data.to_numpy()

    beta1_array = np.linspace(.003, .008, 1000)
    dl = [_dl_ddb(beta1, times_to_catastrophe) for beta1 in beta1_array]
    dl = np.array(dl)
    
    # Create the figure, stored in variable `p`
    p = bokeh.plotting.figure(
        width=400,
        height=300,
        x_axis_label="various values of beta1",
        y_axis_label="log likelihood dl/delta_beta",
        title = 'roots where beta1 maximizes mle',
    )
    p.line(
        beta1_array,
        dl,
        x="beta1",
        y="dl/beta1"
    )
    return p

#uncomment to run
# p = plot_brentq_roots()
# bokeh.io.show(p)

# +
def main():
# #     uncomment to use code
#     df = _create_df_for_dashboard()

#     beta1_slider = pn.widgets.DiscreteSlider(
#         name='beta1',
#         options=list(df["beta1"].unique()),
#         value = 1.0
#     )

#     beta2_slider = pn.widgets.DiscreteSlider(
#         name='beta2',
#         options=list(df["beta2"].unique()),
#         value = 1.0
#     )
#     # to use, uncomment out this code
#     widgets = pn.Column(pn.Spacer(height=30), beta1_slider, beta2_slider, width=200)
#     pn.Row(plot_ecdf_pn, widgets)
    
    p2 = plot_beta_ratios_ecdf()
    bokeh.io.show(p2)
    
    bokeh.io.save(
        p2,
        filename="viz_explore_two_arrival_story_Fig2a.html",
        title="Exploring two arrival story",
    )
    
    p3 = plot_brentq_roots()
    bokeh.io.show(p3)
    
    bokeh.io.save(
        p3,
        filename="viz_explore_two_arrival_story_Fig2b.html",
        title="Exploring two arrival story",
    )
    
    plots = [p2, p3]
    bokeh.io.save(
        plots,
        filename="viz_explore_two_arrival_story_Fig2.html",
        title="Exploring two arrival story",
    )
    
    
    return True

if __name__ == '__main__' : main()

# +
#!jupytext --to python viz_explore_two_arrival_story.ipynb
# -


