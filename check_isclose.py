""" 
This file is used to check that the both matlab and python numpy results are close with small errors allowable to account for floating point errors 
Py_array is a numpy array with its result  generated via python then stored in a csv file titled 
M_array is a numpy array with its result generated via 

"""
from scipy.io import loadmat 
import numpy as np
import pandas as pd

def save_array(data: np.ndarray):
    np.savetxt("numpy_result.csv", data, delimiter=",")


def check_arrays_are_close(numpy_res: np.ndarray, matlab_res:np.ndarray):
    np.allclose(numpy_res, matlab_res, atol=1e-12)

