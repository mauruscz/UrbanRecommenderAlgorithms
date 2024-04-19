import pandas as pd
import pathlib
from tqdm import tqdm

DATASET_NAME = 'foursquare_complete'
userColName = 'uid'
poiColName = 'venue_id'
# metadataColNames = ['category', 'lat', 'lon']
# metadataColTypes = ['token_seq', 'float', 'float']
metadataColNames = ['lat', 'lon']
metadataColTypes = ['float', 'float']


base_dir = pathlib.Path.cwd().parent
data_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed'
processed_dataset_dir = processed_dir / DATASET_NAME
output_dir = base_dir / 'data' / 'output'
dataset_path = data_dir / f'{DATASET_NAME}.csv'

df = pd.read_csv(dataset_path, index_col=None if 'full' in DATASET_NAME or 'complete' in DATASET_NAME else 0)


# First part: interaction file generation
user_remapping, cnt_uid = {}, 0
poi_remapping, cnt_poi = {}, 0
user_id_list = []
poi_id_list = []
timestamp_list = []
for user_id in tqdm(df[userColName].unique(), desc='uid'):
    timestamp = 0
    if user_id in user_remapping:
        new_user_id = user_remapping[user_id]
    else:
        new_user_id = f'{cnt_uid}U'
        user_remapping[user_id] = f'{cnt_uid}U'
        cnt_uid += 1
    for poi_id in df[df[userColName] == user_id][poiColName]:
        if poi_id in poi_remapping:
            new_poi_id = poi_remapping[poi_id]
        else:
            new_poi_id = f'{cnt_poi}I'
            poi_remapping[poi_id] = f'{cnt_poi}I'
            cnt_poi += 1
        user_id_list.append(new_user_id)
        poi_id_list.append(new_poi_id)
        timestamp_list.append(timestamp)
        timestamp += 1

# Create and save interaction file
new_df = pd.DataFrame(data={'user_id:token': user_id_list,
                            'item_id:token': poi_id_list,
                            'timestamp:float': timestamp_list})
processed_dataset_dir.mkdir(exist_ok=True, parents=True)
new_df.to_csv(processed_dataset_dir / f'{DATASET_NAME}.inter', index=False, sep='\t')

poi_id_list = []
poi_metadata_lists = [[] for _ in range(len(metadataColNames))]

# Second part: generate files with user/item metadata
for poi_id in tqdm(df[poiColName].unique(), 'item metadata'):
    poi_id_list.append(poi_remapping[poi_id])
    poi_metadata = df[df[poiColName] == poi_id].iloc[0]
    for i in range(len(metadataColNames)):
        metadataColName = metadataColNames[i]
        poi_metadata_lists[i].append(poi_metadata[metadataColName])
df_dict = {'item_id:token': poi_id_list}
for i in range(len(metadataColNames)):
    metadataColName, metadataColType = metadataColNames[i], metadataColTypes[i]
    df_dict[f'{metadataColName}:{metadataColType}'] = poi_metadata_lists[i]
# Create and save item metadata file
item_df = pd.DataFrame(data=df_dict)
item_df.to_csv(processed_dataset_dir / f'{DATASET_NAME}.item', index=False, sep='\t')

# Compute Explorer Index for each user
uid_list = []
user_metadata_list = []
for user_id in tqdm(df[userColName].unique(), 'user metadata'):
    num_unique_poi = len(set(df[df[userColName] == user_id][poiColName]))
    num_visited_poi = len(df[df[userColName] == user_id][poiColName])
    user_metadata_list.append(num_unique_poi / num_visited_poi)
    uid_list.append(user_remapping[user_id])
# Create and save user metadata file
user_df = pd.DataFrame(data={'user_id:token': uid_list, 'explorer_index:float': user_metadata_list})
user_df.to_csv(processed_dataset_dir / f'{DATASET_NAME}.user', index=False, sep='\t')
