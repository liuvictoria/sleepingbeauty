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
import tidy_data
import iqplot
import numpy as np
import ecdfs

import bebi103

import bokeh.io
bokeh.io.output_notebook()
# -

def plot_ecdf_conf_int_overlap():
    """
    Visualize conf int overlap of catastrophe times,
    between labeled and unlabeled tublin
    Inputs:
        df df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Output:
        p : Bokeh figure
    """
    
    df = tidy_data.tidy_dic()
    # plot ecdfs using iqplot
    p = bokeh.plotting.figure(
        title = 'Catastrophe times for microtubules',
        width = 500,
        height = 400,
        x_axis_label = 'time to catastrophe (s)',
        y_axis_label = 'ECDF',
        tooltips=[
        ("catastrophe time (s)", "@{time to catastrophe (s)}"),
    ],
    )
    
    iqplot.ecdf(
        data = df,
        q = 'time to catastrophe (s)',
        cats = 'tubulin_labeled',
        conf_int = True,
        palette = ['crimson', 'forestgreen'],
        p = p,
        conf_int_kwargs = {"fill_alpha": 0.35,}
        )
    return p
# p = plot_ecdf_conf_int_overlap(df)
# bokeh.io.show(p)

# defining epsilon via alpha
def _epsilon(n, alpha):
    """
    Find epsilon, given alpha and n
    Based on the equation from the homework problem
    """
    return np.sqrt(
        np.log(2 / alpha) / 2 / n
    )

def _bounds(data, alpha):
    """
    Find the lower and upper bounds of confidence interval for our data,
    given alpha.
    L(x) = max(0, ecdf(data, data) - epsilon)
    U(x) = min(1, ecdf(data, data) + epsilon)
    """
    # get parameters
    n = len(data)
    eps = _epsilon(n, alpha)
    
    # calculate bounds
    lower_bounds = np.maximum(np.zeros(n), ecdfs.ecdf(data,data) - eps)
    upper_bounds = np.minimum(np.ones(n), ecdfs.ecdf(data,data) + eps)
    
    return lower_bounds, upper_bounds

def plot_dkw_conf_int_bounds():
    """
    Visualize conf int boundaries for catastrophe time ECDFs,
    between labeled and unlabeled tublin
    Inputs:
        df df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Output:
        p : Bokeh figure
    """
    
    df = tidy_data.tidy_dic()
    
    #Labeled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values

    #Unlabeled data
    unlabeled_data = df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values
    
    # finding the bounds for labeled & unlabeled data
    lower_labeled, upper_labeled = _bounds(labeled_data, 0.05)
    lower_unlabeled, upper_unlabeled = _bounds(unlabeled_data, 0.05)
    

    p = bokeh.plotting.figure(
        title = 'DKW ECDF Confidence Interval Bounds for Catastrophe Times',
        width = 500,
        height = 400,
        x_axis_label = 'time to catastrophe (s)',
        y_axis_label = 'ECDF',
    )
    p.circle(
        labeled_data,
        lower_labeled,
        color='crimson',
        size = 2,
        legend_label = 'labeled tubulin'
    )
    
    p.circle(
        labeled_data,
        upper_labeled,
        color='crimson',
        size = 2,
        legend_label = 'labeled tubulin'
    )
    
    p.circle(
        unlabeled_data,
        lower_unlabeled,
        color='forestgreen',
        size = 2,
        legend_label = 'unlabeled tubulin'
    )

    p.circle(
        unlabeled_data,
        upper_unlabeled,
        color='forestgreen',
        size = 2,
        legend_label = 'unlabeled tubulin'
    )
    p.legend.location = 'bottom_right'
    return p
# df = tidy_data.tidy_dic()
# plot_dkw_conf_int_bounds(df)
# p = plot_dkw_conf_int_bounds(df)
# bokeh.io.show(p)

def plot_dkw_conf_int_bounds_fill_between():
    """
    Visualize conf int boundaries for catastrophe time ECDFs,
    between labeled and unlabeled tublin
    Inputs:
        df df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Output:
        p : Bokeh figure
    """
    df = tidy_data.tidy_dic()
    #Labeled data
    labeled_data = np.sort(
        df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values
    )

    #Unlabeled data
    unlabeled_data = np.sort(
        df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values
    )
    
    # finding the bounds for labeled & unlabeled data
    lower_labeled, upper_labeled = _bounds(labeled_data, 0.05)
    lower_unlabeled, upper_unlabeled = _bounds(unlabeled_data, 0.05)
    
    
    p = bebi103.viz.fill_between(
        labeled_data,
        lower_labeled,
        labeled_data,
        upper_labeled,
        patch_kwargs = {"fill_alpha": 0.6, 'color':'crimson', 'legend_label' : 'labeled'},
        line_kwargs = {'color':'crimson'},
        title = 'DKW ECDF Confidence Interval Bounds for Catastrophe Times',
        x_axis_label = 'time to catastrophe (s)',
        y_axis_label = 'ECDF',
        plot_width = 500, 
        plot_height = 400
    )
    
    bebi103.viz.fill_between(
        unlabeled_data,
        lower_unlabeled,
        unlabeled_data,
        upper_unlabeled,
        patch_kwargs = {"fill_alpha": 0.3, 'color':'forestgreen', 'legend_label' : 'unlabeled'},
        line_kwargs = {'color':'forestgreen'},
        title = 'DKW ECDF Confidence Interval Bounds for Catastrophe Times',
        x_axis_label = 'time to catastrophe (s)',
        y_axis_label = 'ECDF',
        plot_width = 500, 
        plot_height = 400,
        p = p
    )
    

    p.legend.location = 'bottom_right'
    return p
# df = tidy_data.tidy_dic()
# p = plot_dkw_conf_int_bounds_fill_between(df)
# bokeh.io.show(p)

# +
def main():
    p = plot_ecdf_conf_int_overlap()
    bokeh.io.show(p)
    bokeh.io.save(
        p,
        filename="viz_controls_Fig1a.html",
        title="Labeled vs unlabeled catastrophe times",
    )
    
    p3 = plot_dkw_conf_int_bounds_fill_between()
    bokeh.io.show(p3)
    
    bokeh.io.save(
        p3,
        filename="viz_controls_Fig1b.html",
        title="Labeled vs unlabeled catastrophe times",
    )
    
    p2 = plot_dkw_conf_int_bounds()
    bokeh.io.show(p2)
    bokeh.io.save(
        p2,
        filename="viz_controls_Fig1c.html",
        title="Labeled vs unlabeled catastrophe times",
    )
    
    plots = [p, p3, p2]
    bokeh.io.show(bokeh.layouts.gridplot(plots, ncols=1))
    
    bokeh.io.save(
        plots,
        filename="viz_controls_Fig1.html",
        title="Labeled vs unlabeled catastrophe times",
    )
    
    return True
    
if __name__ == '__main__': main()

# +
#!jupytext --to python viz_controls.ipynb
