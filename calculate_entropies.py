import pandas as pd
import numpy as np
import sys
from scipy import stats

from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()

from math import pi, sin, cos, sqrt, atan2

earthradius = 6371.0


def getDistanceByHaversine(loc1, loc2):
    "Haversine formula - give coordinates as (lat_decimal,lon_decimal) tuples"

    lat1, lon1 = loc1
    lat2, lon2 = loc2
    
    # convert to radians
    lon1 = lon1 * pi / 180.0
    lon2 = lon2 * pi / 180.0
    lat1 = lat1 * pi / 180.0
    lat2 = lat2 * pi / 180.0

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2.0))**2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    km = earthradius * c
    return km




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




def _radius_of_gyration_individual(traj):
    """
    Compute the radius of gyration of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectory of the individual.
    
    Returns
    -------
    float
        the radius of gyration of the individual.
    """
    lats_lngs = traj[["lat", "lon"]].values
    center_of_mass = np.mean(lats_lngs, axis=0)
    rg = np.sqrt(np.mean([getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0 for lat, lng in lats_lngs]))
    return rg


def radius_of_gyration(traj, show_progress=True):
    """Radius of gyration.
    
    Compute the radii of gyration (in kilometers) of a set of individuals in a TrajDataFrame.
    The radius of gyration of an individual :math:`u` is defined as [GHB2008]_ [PRQPG2013]_: 
    
    .. math:: 
        r_g(u) = \sqrt{ \\frac{1}{n_u} \sum_{i=1}^{n_u} dist(r_i(u) - r_{cm}(u))^2}
    
    where :math:`r_i(u)` represents the :math:`n_u` positions recorded for :math:`u`, and :math:`r_{cm}(u)` is the center of mass of :math:`u`'s trajectory. In mobility analysis, the radius of gyration indicates the characteristic distance travelled by :math:`u`.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the radius of gyration of each individual.
    
    """
    # if 'uid' column in not present in the TrajDataFrame
    if "uid" not in traj.columns:
        return pd.DataFrame([_radius_of_gyration_individual(traj)], columns=[sys._getframe().f_code.co_name])
    
    if show_progress:
        df = traj.groupby("uid").progress_apply(lambda x: _radius_of_gyration_individual(x))
    else:
        df = traj.groupby("uid").apply(lambda x: _radius_of_gyration_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name}) 
