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
package with ecdf utilities

Functions:
    ecdf_vals(data)
    ecdf(x, data)

The difference between the two functions is that
ecdf_vals only ranks observed data, while ecdf
ranks any x, given the data
"""

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 986}, "id": "3CUu8LKsIAAd", "outputId": "6cc38367-3faa-41fe-a10e-35fbfd148537"}
import numpy as np
import math

# + {"id": "SrnSmF6qIAAq"}
def ecdf_vals(data):
    '''
    Calculates the empirical distribution function (ECDF) for a given 
    data set. In other words, finds the proportion of values which are
    less than or equal to the given data point. 
    
    If two values in the data are the same, will return evenly-spaced 
    points for each of these values. (When plotted, these will create a
    vertical line in the ECDF as expected.)
    
    Inputs:
        data : list, tuple, or ndarray of ints or floats
            Single-dimensional, numerical data for which the ECDF 
            is desired.
    Outputs: 
        x : ndarray
            The original data, sorted using np.sort()
        y : ndarray
            Values of the ECDF corresponding pairwise to the data in x.   
    
    '''
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y
# -

def _find_nearest(array, value):
    """
    Given an array and a value, which index of the array contains the value to which
    the input value is closest?
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def ecdf(x, data):
    '''
    Calculates the empirical distribution function (ECDF) for a given 
    data set 'data' and returns what the percentile for 'x', given this
    ECDF. In other words, finds the proportion of values which are
    less than or equal to 'x', based on 'data'. 
    
    Inputs:
           x : list, tuple, or ndarray of ints or floats
        data : list, tuple, or ndarray of ints or floats
            Single-dimensional, numerical data for which the ECDF 
            is desired.
    Outputs: 
        y : ndarray
            Value(s) of the ECDF corresponding to x, given ECDF of 'data'.   
    
    '''
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_x = []
    for elem in x:
        #return 0 for 'x' values below the min value of 'data'
        if elem < np.min(data):
            cdf_x.append(0)
        else:
            #using helper function to return index of closest element
            idx = _find_nearest(sorted_data, elem)
            cdf_x.append(y[idx])
    return cdf_x

# +
#!jupytext --to python ecdfs.ipynb
