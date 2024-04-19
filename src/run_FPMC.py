# Hyper-parameters
DATASET_NAME = 'foursquare_complete'
MODEL_NAME = "FPMC"
device_id = '3'
epochs = 100
early_stopping = 5
embedding_size = 16
min_items_occurrences, min_users_interactions = 5, 5
topk_evalgrid = [10, 20, 50, 100]

import os

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = device_id
import pathlib
import pickle
import numpy as np
import pandas as pd

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, LightSANs, STAMP, FPMC
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, ModelType
from tqdm import tqdm

import utils

base_dir = pathlib.Path.cwd().parent
output_data_dir = base_dir / 'data' / 'output' / DATASET_NAME / MODEL_NAME
output_data_dir.mkdir(parents=True, exist_ok=True)

params_dict = {'gpu_id': device_id,
               'use_gpu': len(device_id) > 0,
               'dataset': DATASET_NAME,
               'MODEL_TYPE': ModelType.SEQUENTIAL,
               'data_path': str(base_dir / 'data' / 'processed'),
               'USER_ID_FIELD': 'user_id',
               'ITEM_ID_FIELD': 'item_id',
               'TIME_FIELD': 'timestamp',
               'user_inter_num_interval': f"[{min_users_interactions},inf)",
               'item_inter_num_interval': f"[{min_items_occurrences},inf)",
               'MAX_ITEM_LIST_LENGTH': 1500,
               'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
               'loss_type': 'BPR',
               'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 1},
               'epochs': epochs,
               'stopping_step': early_stopping,
               'embedding_size': embedding_size,
               'eval_args': {
                   'split': {'LS': 'valid_and_test'},
                   'group_by': 'user',
                   'order': 'TO',
                   'mode': 'pop100',
                   'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
                   'topk': [10],
                   'valid_metric': 'Hit@10',
                   'repeatable': True,
                   'eval_batch_size': 4096
               }}

# configurations initialization
config = Config(model=MODEL_NAME, config_dict=params_dict)

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()

# write config info into log
logger.info(config)

# dataset creating and filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
print('Dataset preparation...')
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
if MODEL_NAME == 'GRU4Rec':
    model = GRU4Rec(config, train_data.dataset).to(config['device'])
elif MODEL_NAME == 'LightSANs':
    model = LightSANs(config, train_data.dataset).to(config['device'])
elif MODEL_NAME == 'STAMP':
    model = STAMP(config, train_data.dataset).to(config['device'])
elif MODEL_NAME == 'FPMC':
    model = FPMC(config, train_data.dataset).to(config['device'])
else:
    raise Exception(f'{MODEL_NAME} is not supported.')
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
print('Start training...')
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# model evaluation
test_result = trainer.evaluate(test_data)
print(test_result)

# Compute popularity of each item
item_pop = np.full(fill_value=0, shape=(dataset.item_num - 1,))
item_id_list = list(dataset.item_counter.keys())
for item_id in tqdm(item_id_list, 'get item pop'):
    item_pop[item_id - 1] = dataset.item_counter[item_id]

print('Start Evaluation...')
# Get predictions on the test set
for topk in tqdm(topk_evalgrid, 'topk eval'):
    # Bias towards users
    user_metadata = pd.read_csv(base_dir / 'data' / 'processed' / DATASET_NAME / f'{DATASET_NAME}.user', sep='\t')
    user_metadata_rnd_entropy = pd.read_csv(base_dir / 'data' / 'output' / f'{DATASET_NAME}_random_entropy.csv',
                                            sep=',')
    user_metadata_rnd_entropy['uid'] = user_metadata_rnd_entropy['uid'].apply(lambda x: f'{x - 1}U')
    user_metadata_uncorrelated_entropy = pd.read_csv(
        base_dir / 'data' / 'output' / f'{DATASET_NAME}_uncorrelated_entropy.csv', sep=',')
    user_metadata_uncorrelated_entropy['uid'] = user_metadata_uncorrelated_entropy['uid'].apply(lambda x: f'{x - 1}U')
    user_hit_list = []
    # Bias towards items
    hitrate_by_items, occurrences_by_items = np.zeros(dataset.item_num - 1), np.zeros(dataset.item_num - 1)
    all_predicted_items = []
    for userid in tqdm(user_metadata['user_id:token'], 'users'):
        trainer.model.eval()
        ground_truth_item = utils.get_last_item(dataset, userid)
        rec_list = utils.predict_for_all_item_fpmc(userid, dataset, trainer.model, config, test_data,
                                                   topk).indices.detach().cpu().numpy()[0]
        # User Bias
        user_hit_list.append(ground_truth_item in rec_list)
        # Item Bias
        hitrate_by_items[ground_truth_item - 1] += ground_truth_item in rec_list
        occurrences_by_items[ground_truth_item - 1] += 1
        all_predicted_items += rec_list[0]
        all_predicted_items = list(set(all_predicted_items))
    # User Bias
    user_metadata['hit'] = user_hit_list
    user_metadata_rnd_entropy['hit'] = user_hit_list
    user_metadata_uncorrelated_entropy['hit'] = user_hit_list
    user_metadata.to_csv(output_data_dir / f'hit@{topk}_by_EI.csv', sep=',')
    user_metadata_rnd_entropy.to_csv(output_data_dir / f'hit@{topk}_by_rnd_entropy.csv', sep=',')
    user_metadata_uncorrelated_entropy.to_csv(output_data_dir / f'hit@{topk}_by_uncorrelated_entropy.csv', sep=',')
    # Item Bias
    coverage = len(set(all_predicted_items)) / (dataset.item_num - 1)
    hitrate_by_items = hitrate_by_items / occurrences_by_items
    pd.DataFrame(data={'HitRate': hitrate_by_items, 'ItemPop': item_pop}).to_csv(
        output_data_dir / f'hit@{topk}_by_itempop.csv', sep=',')
    with open(output_data_dir / f'coverage@{topk}.pkl', 'wb') as file:
        pickle.dump(coverage, file)
    print(f'--- TEST METRICS k={topk} ---')
    print(f'HitRate@{topk}: {round(np.nanmean(hitrate_by_items), 4)}')
    print(f'Coverage@{topk}: {round(coverage, 4)}')
