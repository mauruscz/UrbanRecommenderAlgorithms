import pathlib
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import utils

DATASET_NAME = 'foursquare_complete'
userColName = 'uid'
poiColName = 'venue_id'
metadataColNames = ['lat', 'lon']
metadataColTypes = ['float', 'float']

base_dir = pathlib.Path.cwd().parent
data_dir = base_dir / 'data' / 'raw'
output_data_dir = base_dir / 'data' / 'output'

dataset_path = data_dir / f'{DATASET_NAME}.csv'

df = pd.read_csv(dataset_path, index_col=None if 'full' in DATASET_NAME or 'complete' in DATASET_NAME else 0)
re = utils.random_entropy(df)
ue = utils.uncorrelated_entropy(df)

# plot the distribution of the uncorrelated entropy, increase the figsize if needed
plt.figure(figsize=(10, 6))
plt.hist(ue["uncorrelated_entropy"], bins=100)
plt.xlabel("Uncorrelated entropy")
plt.ylabel("Frequency")
plt.title("Distribution of the uncorrelated entropy")
plt.show()

# do the same for the random entropy
plt.figure(figsize=(10, 6))
plt.hist(re["random_entropy"], bins=100)
plt.xlabel("Random entropy")
plt.ylabel("Frequency")
plt.title("Distribution of the random entropy")
plt.show()

re.to_csv(output_data_dir / f"{DATASET_NAME}_random_entropy.csv", index=False)
ue.to_csv(output_data_dir / f"{DATASET_NAME}_uncorrelated_entropy.csv", index=False)
