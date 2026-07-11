import os
import numpy as np
import pandas as pd

__dir__ = os.path.abspath(os.path.dirname(__file__))

def load_fennell2023():
    '''
    Load data from experiment one in Fennell and Ratcliff (2023).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data with columns: 'subjectNumber', 'blockNumber', 'trialNumber', 'rt', 'numberOfStimulus', 'responseError'
    '''

    data_path = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "data", "Fennell2023_exp4.csv")

    data = pd.read_csv(data_path, index_col=0)
    return data


def load_kvam2019():
    '''
    Load data from Kvam et al. (2019).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data with columns: 
    '''

    data_path = os.path.join(os.path.dirname(os.path.dirname(__dir__)), "data", "Kvam2019.csv")

    data = pd.read_csv(data_path)

    data = data.drop(columns=['isCued', 'cueOrientation', 
                                'cueDeflections', 'points', 
                                'jitterLevel', 'targetOrientation'])
    data = data.rename(columns={'RT':'rt'})
    data.deviation *= 2
    data.absoluteDeviation = np.abs(data.deviation)
    return data