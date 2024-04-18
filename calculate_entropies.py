import pandas as pd
import numpy as np
import sys
from scipy import stats

from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()


def _random_entropy_individual(traj):
    """
    Compute the random entropy of a single individual given their TrajDataFrame.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individual
    
    Returns
    -------
    float
        the random entropy of the individual 
    """
    n_distinct_locs = len(traj.groupby(["lat", "lon"]))
    entropy = np.log2(n_distinct_locs)
    return entropy


def random_entropy(traj, show_progress=True):
    """Random entropy.
    
    Compute the random entropy of a set of individuals in a TrajDataFrame.
    The random entropy of an individual :math:`u` is defined as [EP2009]_ [SQBB2010]_: 
    
    .. math::
        E_{rand}(u) = log_2(N_u)
    
    where :math:`N_u` is the number of distinct locations visited by :math:`u`, capturing the degree of predictability of :math:`u`’s whereabouts if each location is visited with equal probability. 

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the random entropy of the individuals.
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_random_entropy_individual(traj)], columns=[sys._getframe().f_code.co_name])
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _random_entropy_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _random_entropy_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})







def _uncorrelated_entropy_individual(traj, normalize=False):
    """
    Compute the uncorrelated entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    n = len(traj)
    probs = [1.0 * len(group) / n for group in traj.groupby(by=["lat", "lon"]).groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(traj[["lat", "lon"]].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def uncorrelated_entropy(traj, normalize=False, show_progress=True):
    """Uncorrelated entropy.
    
    Compute the temporal-uncorrelated entropy of a set of individuals in a TrajDataFrame. The temporal-uncorrelated entropy of an individual :math:`u` is defined as [EP2009]_ [SQBB2010]_ [PVGSPG2016]_: 
    
    .. math::
        E_{unc}(u) = - \sum_{j=1}^{N_u} p_u(j) log_2 p_u(j)
    
    where :math:`N_u` is the number of distinct locations visited by :math:`u` and :math:`p_u(j)` is the historical probability that a location :math:`j` was visited by :math:`u`. The temporal-uncorrelated entropy characterizes the heterogeneity of :math:`u`'s visitation patterns.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated entropy of the individuals.
    
    """
    column_name = sys._getframe().f_code.co_name
    if normalize:
        column_name = 'norm_%s' % sys._getframe().f_code.co_name
    
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_uncorrelated_entropy_individual(traj)], columns=[column_name])
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    else:
        df = traj.groupby("uid").apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    return pd.DataFrame(df).reset_index().rename(columns={0: column_name})

