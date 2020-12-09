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

import tidy_data
import numpy as np
import bebi103
import scipy

# +
def conf_int_means(df):
    """
    get 95% confidence intervals for labeled and unlabeled 
    mean catastrophe times
    
    Inputs:
        df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Outputs:
        (
        bs_unlabeled_mean_conf, 
        bs_labeled_mean_conf
        )
    """
    #Labeled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values

    #Unlabeled data
    unlabeled_data = df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values
    
    # get bootstrap replicates
    bs_labeled_means = bebi103.bootstrap.draw_bs_reps(labeled_data, np.mean, size=10000)
    bs_unlabeled_means = bebi103.bootstrap.draw_bs_reps(unlabeled_data, np.mean, size=10000)

    # calculate 95% confidence interval for labeled
    bs_labeled_mean_conf = np.percentile(bs_labeled_means, [2.5, 97.5])

    # calculate 95% confidence interval for unlabeled
    bs_unlabeled_mean_conf = np.percentile(bs_unlabeled_means, [2.5, 97.5])
    
    return (bs_unlabeled_mean_conf, bs_labeled_mean_conf)

#output
# Mean unlabeled conf int: [353.105, 477.475]
# Mean labeled conf int: [402.275, 481.303]

# +
def diff_means(df):
    """
    get test statistic and and p-value for 
    difference of mean catastrophe times.
    Null hypothesis that the two distributions are the same
    and thus the means are also the same
    
    Inputs:
        df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Outputs:
        (diff_mean, p_val)
    """
    
    #Labeled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values

    #Unlabeled data
    unlabeled_data = df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values
    
    # Compute test statistic for original data set
    diff_mean = np.mean(labeled_data) - np.mean(unlabeled_data)

    # Draw permutation replicates
    perm_reps = bebi103.bootstrap.draw_perm_reps(
        labeled_data, unlabeled_data, bebi103.bootstrap.diff_of_means, size = 10000
    )

    # Compute p-value
    p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)
    return (diff_mean, p_val)

#Output
# Experimental difference of means: 28.185
# p-value = 0.229

# +
def diff_means_student(df):
    """
    get test statistic and and p-value for 
    difference of mean catastrophe times, assuming
    student t distribution. Takes into account std
    
    Null hypothesis that the two distributions are the same
    and thus the means are also the same
    
    Inputs:
        df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Outputs:
        (diff_mean_studentized, p_val_studentized)
    Notes: 
        Uses bebi103.bootstrap.studentized_diff_of_means
    """
    
    #Labeled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values

    #Unlabeled data
    unlabeled_data = df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values
    
    diff_mean_studentized = bebi103.bootstrap.studentized_diff_of_means(
        labeled_data, unlabeled_data
    )

    # Draw permutation replicates
    perm_reps_studentized = bebi103.bootstrap.draw_perm_reps(
        labeled_data, unlabeled_data, 
        bebi103.bootstrap.studentized_diff_of_means, size = 10000
    )

    # Compute p-value
    p_val_studentized = np.sum(
        perm_reps_studentized >= diff_mean_studentized
        ) / len(perm_reps_studentized)
    
    return (diff_mean_studentized, p_val_studentized)

#Output
# Experimental studentized difference of means: 0.752
# p-value = 0.239


# +
def conf_int_means_normal(df):
    """
    get 95% confidence intervals for labeled and unlabeled 
    mean catastrophe times assuming normal distribution;
    interval over which 95% of the probability mass of the normal distribution lies 
    
    Inputs:
        df : dataframe of gardner_time_to_catastrophe_dic_tidy.csv dataset
    Outputs:
        (
        conf_int_unlabeled,
        conf_int_labeled
        )
    """
    #Labeled data
    labeled_data = df.loc[df['labeled'] == True, 'time to catastrophe (s)'].values

    #Unlabeled data
    unlabeled_data = df.loc[df['labeled'] == False, 'time to catastrophe (s)'].values

    # mean of labeled
    labeled_mean = np.mean(labeled_data)

    # mean of unlabled
    unlabeled_mean = np.mean(unlabeled_data)

    # CI of labeled
    conf_int_labeled = scipy.stats.norm.interval(0.95, loc=labeled_mean, scale=np.std(labeled_data)/np.sqrt(len(labeled_data)))
    
    # CI of unlabeled
    conf_int_unlabeled = scipy.stats.norm.interval(0.95, loc=unlabeled_mean, scale=np.std(unlabeled_data)/np.sqrt(len(unlabeled_data)))
    
    return (conf_int_unlabeled, conf_int_labeled)

# #output
# Mean unlabeled conf int: [351.165, 473.888]
# Mean labeled conf int: [400.836, 480.586]

# +
def main():
    df = tidy_data.tidy_dic()
    bs_unlabeled_mean_conf, bs_labeled_mean_conf = conf_int_means(df)

    print(
    """
    Mean unlabeled conf int: [{0:.3f}, {1:.3f}]
    """.format(
            *tuple(bs_unlabeled_mean_conf)
        )
    )
    
    print(
        """
    Mean labeled conf int: [{0:.3f}, {1:.3f}]
    """.format(
            *tuple(bs_labeled_mean_conf)
        )
    )
    
    print('\n\n')
    
    diff_mean, p_val = diff_means(df)
    
    print(f'Experimental difference of means: {diff_mean:.3f}')
    print(f'p-value = {p_val:.3f}')
    
    print('\n\n')
    
    diff_mean_studentized, p_val_studentized = diff_means_student(df)
    
    print(f'Experimental studentized difference of means: {diff_mean_studentized:.3f}')
    print(f'p-value = {p_val_studentized:.3f}')
    
    print('\n\n')
    
    normal_conf_int_unlabeled, normal_conf_int_labeled = conf_int_means_normal(df)
    
    print(
    """
    Mean unlabeled conf int: [{0:.3f}, {1:.3f}]
    """.format(
            *tuple(normal_conf_int_unlabeled)
        )
    )
    
    print(
    """
    Mean labeled conf int: [{0:.3f}, {1:.3f}]
    """.format(
            *tuple(normal_conf_int_labeled)
        )
    )
    return True
    
if __name__ == '__main__': main()

# +
#!jupytext --to python controls.ipynb
