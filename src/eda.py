import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pathlib

DATASET_NAME = 'foursquare_complete'
userColName = 'uid'
poiColName = 'venue_id'  # poi venue_id

base_dir = pathlib.Path.cwd().parent
data_dir = base_dir / 'data' / 'raw'
output_dir = base_dir / 'data' / 'output'
dataset_path = data_dir / f'{DATASET_NAME}.csv'

df = pd.read_csv(dataset_path, index_col=None if 'full' in DATASET_NAME or 'complete' in DATASET_NAME else 0)
# Question 1: Are there repetitions in the POI sequence of each user?
count_repetitions = 0
for user_id in df[userColName].unique():
    count_repetitions += len(df[df[userColName] == user_id][poiColName]) != len(set(df[df[userColName] == user_id][poiColName]))
print(f'{count_repetitions} users out of {df[userColName].nunique()} revisits more than once at least a POI')
# Question 2: How many POI each user revisits? Are there different users who tend to revisit more than others?
count_uniquepois_list = []
sequence_length_list = []
for user_id in df[userColName].unique():
    sequence_length_list.append(len(df[df[userColName] == user_id][poiColName]))
    count_uniquepois_list.append(len(set(df[df[userColName] == user_id][poiColName])))
count_uniquepois_list = np.array(count_uniquepois_list)
sequence_length_list = np.array(sequence_length_list)
explorer_index_list = count_uniquepois_list / sequence_length_list
print(f'EI mean: {round(explorer_index_list.mean(), 2)}, std: {round(explorer_index_list.std(), 2)}')
print(f'EI max: {round(explorer_index_list.max(), 2)}, min: {round(explorer_index_list.min(), 2)}')
# Question 3: What is the user and item distribution? Does it follow a long-tail as in usual recommender scenarios?
## USERS
# user_length_values, user_length_counts = np.unique(sequence_length_list, return_counts=True)
_, user_length_counts = np.unique(df[userColName].values, return_counts=True)
user_length_values, user_length_counts = np.unique(user_length_counts, return_counts=True)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(range(len(user_length_counts)),
            user_length_counts, s=10)
ax.set_xlabel('User Degree', fontsize=30)
ax.set_ylabel('Frequency', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(output_dir / f'{DATASET_NAME}_user_distribution.png')
fig.savefig(output_dir / f'{DATASET_NAME}_user_distribution.pdf')
## ITEMS
_, item_length_counts = np.unique(df[poiColName].values, return_counts=True)
item_length_values, item_length_counts = np.unique(item_length_counts, return_counts=True)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(range(len(item_length_counts)),
            item_length_counts, s=10)
ax.set_xlabel('POI Degree', fontsize=30)
ax.set_ylabel('Frequency', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(output_dir / f'{DATASET_NAME}_item_distribution.png')
fig.savefig(output_dir / f'{DATASET_NAME}_item_distribution.pdf')
