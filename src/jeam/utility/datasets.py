import os
import numpy as np
import pandas as pd
from importlib import resources

__dir__ = os.path.abspath(os.path.dirname(__file__))

def _read_csv(filename, **kwargs):
    resource = resources.files("jeam").joinpath("data", filename)
    with resources.as_file(resource) as data_path:
        return pd.read_csv(data_path, **kwargs)

def load_fennell2023():
    '''
    Load data from experiment one in Fennell and Ratcliff (2023).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data with columns: 'subjectNumber', 'blockNumber', 'trialNumber', 'rt', 'numberOfStimulus', 'responseError'
    '''
    return _read_csv("Fennell2023_exp4.csv", index_col=0)


def load_kvam2019():
    '''
    Load data from Kvam et al. (2019).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the data with columns: 
    '''
    data = _read_csv("Kvam2019.csv")
    data = data.drop(
        columns=[
            "isCued",
            "cueOrientation",
            "cueDeflections",
            "points",
            "jitterLevel",
            "targetOrientation",
        ]
    )
    data = data.rename(columns={"RT": "rt"})
    data["deviation"] *= 2
    data["absoluteDeviation"] = np.abs(data["deviation"])
    return data