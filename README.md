# Microtubule Catastrophe
In this repository, we model microtubule catastrophe times with two different models, at five different concentrations. The repo contains the data (graciously provided by [Gardner et al](https://cbs.umn.edu/gardner-lab/home)), code, figures, and analysis for BeBi103 Fall 2020 @Caltech. This is branch containing the actual package, and should be pip-able.

The website exists :crown:[here](https://liuvictoria.github.io/sleepingbeauty/):crown:.


The team consists of Victoria Liu, Sara Adams, Ruby Cheetham, and Makayla Betts.


This repository is called sleepingbeauty because Aurora also had a spindle catastrophe :crown: :european_castle: :dizzy: :sleeping: :eyes: :dancer:

## `microtubulepkg`
Python scripts with all the functions we used to analyze and plot the data


### tidy_data
Tidies both csv files into pd df (refer to the `data` section)

### ecdfs
Two ecdf functions

### controls
All non-plotting functions in determining whether unlabeled vs labled were comparable

### parameter_estimates
All non-plotting functions to maximize mle for alpha/ beta and beta_1/beta_2 for gamma and two-story distributions, respectively

### viz_controls
Unlabeled vs labeled visualization

### viz_explore_two_arrival_story
Get graphical intuition about two arrival story

### viz_parameter_estimates
Visualize parameter estimates (conf ints and corner plot conf ints)

### viz_explore_concentration_datset
Exploratory; visualize all the different concentrations and  catastrophe time

### viz_model_comparison
Graphs (predictive ecdfs, qq plots) for gamma vs two-events

### viz_concentration_effects
Visualization of beta vs alpha changes when concentration changes

### viz_dashboard
Interactive dashboard with concentration, alpha, and beta selection for catastrophe times



## Acknowledgments

We would like to thank Gardner et al. for making their hard-earned data public so that we could explore and visualize it. We are also indebted to the bebi103 course staff for answering all our questions and giving us support throughout the term. Next, huge shoutout to Griffin Chure for this [awesome reproducible website template](https://github.com/gchure/reproducible_website) and the open source community in general for answering our code-related questions. Finally, we would like to thank Justin Bois for creating this fun, occasionally frustrating, but highly rewarding class.
