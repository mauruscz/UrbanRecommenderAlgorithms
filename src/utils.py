import pandas as pd
import numpy as np
import sys
import torch
from scipy import stats

from tqdm import tqdm, tqdm_notebook
from recbole.data.interaction import Interaction

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


def add_last_item(old_interaction, last_item_id, max_len=50):
    new_seq_items = old_interaction['item_id_list'][-1]
    if old_interaction['item_length'][-1].item() < max_len:
        new_seq_items[old_interaction['item_length'][-1].item()] = last_item_id
    else:
        new_seq_items = torch.roll(new_seq_items, -1)
        new_seq_items[-1] = last_item_id
    return new_seq_items.view(1, len(new_seq_items))


def predict_for_all_item(external_user_id, dataset, model, config, test_data, topk):
    model.eval()
    with torch.no_grad():
        uid_series = dataset.token2id(dataset.uid_field, [external_user_id])
        index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
        input_interaction = dataset[index]
        test = {
            'item_id_list': add_last_item(input_interaction,
                                          input_interaction['item_id'][-1].item(), model.max_seq_length),
            'item_length': torch.tensor(
                [input_interaction['item_length'][-1].item() + 1
                 if input_interaction['item_length'][-1].item() < model.max_seq_length else model.max_seq_length])
        }
        new_inter = Interaction(test)
        new_inter = new_inter.to(config['device'])
        new_scores = model.full_sort_predict(new_inter)
        new_scores = new_scores.view(-1, test_data.dataset.item_num)
        new_scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    return torch.topk(new_scores, topk)


def predict_for_all_item_fpmc(external_user_id, dataset, model, config, test_data, topk):
    model.eval()
    with torch.no_grad():
        uid_series = dataset.token2id(dataset.uid_field, [external_user_id])
        index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
        input_interaction = dataset[index]
        test = {
            'user_id': torch.tensor(uid_series),
            'item_id_list': add_last_item(input_interaction,
                                          input_interaction['item_id'][-1].item(), model.max_seq_length),
            'item_length': torch.tensor(
                [input_interaction['item_length'][-1].item() + 1
                 if input_interaction['item_length'][-1].item() < model.max_seq_length else model.max_seq_length])
        }
        new_inter = Interaction(test)
        new_inter = new_inter.to(config['device'])
        new_scores = model.full_sort_predict(new_inter)
        new_scores = new_scores.view(-1, test_data.dataset.item_num)
        new_scores[:, 0] = -np.inf  # set scores of [pad] to -inf
    return torch.topk(new_scores, topk)


def get_last_item(dataset, userid):
    uid_series = dataset.token2id(dataset.uid_field, [userid])
    index = np.isin(dataset[dataset.uid_field].numpy(), uid_series)
    input_interaction = dataset[index]
    return input_interaction['item_id'][-1].item()
