# -*- coding: utf-8 -*-
"""
Statistics-related functions
"""
# External
import numpy as np

# Original Signal
def get_rel_freqs(array):
    """
    Compute relative frequencies of unique values in array
    
    Parameters
    ----------
    array : decimal, [N, M, ...]
        Input array

    Returns
    -------
    unique_vals : decimal, [L,]
        Unique values, sorted in ascending order

    rel_freqs : decimal, [l,]
        Relative frequency of each unique value
    """
    # Sorting array allows the start indices to indicate how many of each value exists
    unique_vals, start_inds = np.unique(np.sort(array.flatten()), return_index=True)
    n_samps = array.size
    
    # Insert needed since diff() shrinks data by 1
    end_inds = np.concatenate((start_inds, np.array([n_samps])), axis=0)
    rel_freqs = np.diff(end_inds) / n_samps
    
    return unique_vals, rel_freqs
