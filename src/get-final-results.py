import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# Hyper-parameters
DATASET_NAME = 'foursquare_complete'
MODEL_NAME = "FPMC"
topk_evalgrid = [10, 20, 50, 100]

base_dir = pathlib.Path.cwd().parent
output_data_dir = base_dir / 'data' / 'output' / DATASET_NAME / MODEL_NAME

for topk in topk_evalgrid:
    user_metadata = pd.read_csv(output_data_dir / f'hit@{topk}_by_EI.csv', sep=',', index_col=0)
    user_metadata_rnd_entropy = pd.read_csv(output_data_dir / f'hit@{topk}_by_rnd_entropy.csv', sep=',', index_col=0)
    user_metadata_uncorrelated_entropy = pd.read_csv(output_data_dir / f'hit@{topk}_by_uncorrelated_entropy.csv', sep=',', index_col=0)
    item_metadata = pd.read_csv(output_data_dir / f'hit@{topk}_by_itempop.csv', sep=',', index_col=0)
    # Analysis with respect to Explorer Index
    stats_colname = 'explorer_index:float'
    first_quartile_th = np.percentile(user_metadata[stats_colname], 25)
    second_quartile_th = np.percentile(user_metadata[stats_colname], 50)
    third_quartile_th = np.percentile(user_metadata[stats_colname], 75)

    group_list = []
    for index_val in user_metadata[stats_colname]:
        if index_val <= first_quartile_th:
            group_list.append('0')
        elif index_val > first_quartile_th and index_val <= second_quartile_th:
            group_list.append('1')
        elif index_val > second_quartile_th and index_val <= third_quartile_th:
            group_list.append('2')
        else:
            group_list.append('3')
    user_metadata['user_group'] = group_list
    fig, ax = plt.subplots()
    user_metadata['hit'] = user_metadata['hit'].astype(float)
    sns.barplot(x='user_group', y='hit', data=user_metadata, ax=ax, order=['0', '1', '2', '3'])
    ax.set_xticklabels([r'$1^{st}$ quartile', r'$2^{nd}$ quartile', r'$3^{rd}$ quartile', r'$4^{th}$ quartile'],
                       rotation=25)
    ax.set_ylabel(f'HitRate@{topk}', fontsize=26)
    ax.set_xlabel('Explorer Index', fontsize=26)
    ax.set_ylim(0.0, 1.025)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    fig.savefig(output_data_dir / f'barplot_EI@{topk}.png', dpi=500)
    fig.savefig(output_data_dir / f'barplot_EI@{topk}.pdf')

    # Analysis with respect to Predictability (rnd)
    stats_colname = 'random_entropy'
    first_quartile_th = np.percentile(user_metadata_rnd_entropy[stats_colname], 25)
    second_quartile_th = np.percentile(user_metadata_rnd_entropy[stats_colname], 50)
    third_quartile_th = np.percentile(user_metadata_rnd_entropy[stats_colname], 75)

    group_list = []
    for index_val in user_metadata_rnd_entropy[stats_colname]:
        if index_val <= first_quartile_th:
            group_list.append('0')
        elif index_val > first_quartile_th and index_val <= second_quartile_th:
            group_list.append('1')
        elif index_val > second_quartile_th and index_val <= third_quartile_th:
            group_list.append('2')
        else:
            group_list.append('3')
    user_metadata_rnd_entropy['user_group'] = group_list
    fig, ax = plt.subplots()
    user_metadata_rnd_entropy['hit'] = user_metadata_rnd_entropy['hit'].astype(float)
    sns.barplot(x='user_group', y='hit', data=user_metadata_rnd_entropy, ax=ax, order=['0', '1', '2', '3'])
    ax.set_xticklabels([r'$1^{st}$ quartile', r'$2^{nd}$ quartile', r'$3^{rd}$ quartile', r'$4^{th}$ quartile'],
                       rotation=25)
    ax.set_ylabel(f'HitRate@{topk}', fontsize=26)
    ax.set_xlabel('Random Entropy Index', fontsize=26)
    ax.set_ylim(0.0, 1.025)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    fig.savefig(output_data_dir / f'barplot_rnd_PI@{topk}.png', dpi=500)
    fig.savefig(output_data_dir / f'barplot_rnd_PI@{topk}.pdf')

    # Analysis with respect to Predictability (uncorrelated)
    stats_colname = 'uncorrelated_entropy'
    first_quartile_th = np.percentile(user_metadata_uncorrelated_entropy[stats_colname], 25)
    second_quartile_th = np.percentile(user_metadata_uncorrelated_entropy[stats_colname], 50)
    third_quartile_th = np.percentile(user_metadata_uncorrelated_entropy[stats_colname], 75)

    group_list = []
    for index_val in user_metadata_uncorrelated_entropy[stats_colname]:
        if index_val <= first_quartile_th:
            group_list.append('0')
        elif index_val > first_quartile_th and index_val <= second_quartile_th:
            group_list.append('1')
        elif index_val > second_quartile_th and index_val <= third_quartile_th:
            group_list.append('2')
        else:
            group_list.append('3')
    user_metadata_uncorrelated_entropy['user_group'] = group_list
    fig, ax = plt.subplots()
    user_metadata_uncorrelated_entropy['hit'] = user_metadata_uncorrelated_entropy['hit'].astype(float)
    sns.barplot(x='user_group', y='hit', data=user_metadata_uncorrelated_entropy, ax=ax, order=['0', '1', '2', '3'])
    ax.set_xticklabels([r'$1^{st}$ quartile', r'$2^{nd}$ quartile', r'$3^{rd}$ quartile', r'$4^{th}$ quartile'],
                       rotation=25)
    ax.set_ylabel(f'HitRate@{topk}', fontsize=26)
    ax.set_xlabel('Uncorrelated Entropy Index', fontsize=26)
    ax.set_ylim(0.0, 1.025)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    fig.savefig(output_data_dir / f'barplot_uncorrelated_PI@{topk}.png', dpi=500)
    fig.savefig(output_data_dir / f'barplot_uncorrelated_PI@{topk}.pdf')

    # Analysis with respect to Item Popularity
    stats_colname = 'ItemPop'
    item_metadata = item_metadata[~item_metadata['HitRate'].isna()]
    first_quartile_th = np.percentile(item_metadata[stats_colname], 25)
    second_quartile_th = np.percentile(item_metadata[stats_colname], 50)
    third_quartile_th = np.percentile(item_metadata[stats_colname], 75)

    group_list = []
    for index_val in item_metadata[stats_colname]:
        if index_val <= first_quartile_th:
            group_list.append('0')
        elif index_val > first_quartile_th and index_val <= second_quartile_th:
            group_list.append('1')
        elif index_val > second_quartile_th and index_val <= third_quartile_th:
            group_list.append('2')
        else:
            group_list.append('3')
    item_metadata['item_group'] = group_list
    fig, ax = plt.subplots()
    item_metadata['HitRate'] = item_metadata['HitRate'].astype(float)
    sns.barplot(x='item_group', y='HitRate', data=item_metadata, ax=ax, order=['0', '1', '2', '3'])
    ax.set_xticklabels([r'$1^{st}$ quartile', r'$2^{nd}$ quartile', r'$3^{rd}$ quartile', r'$4^{th}$ quartile'],
                       rotation=25)
    ax.set_ylabel(f'HitRate@{topk}', fontsize=26)
    ax.set_xlabel('Item Popularity', fontsize=26)
    ax.set_ylim(0.0, 1.025)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    fig.savefig(output_data_dir / f'barplot_ItemPop@{topk}.png', dpi=500)
    fig.savefig(output_data_dir / f'barplot_ItemPop@{topk}.pdf')

