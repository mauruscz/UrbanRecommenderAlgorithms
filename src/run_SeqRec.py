import pathlib
import pickle
import numpy as np
import pandas as pd

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec, LightSANs, STAMP
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, ModelType
from recbole.utils.case_study import full_sort_topk

DATASET_NAME = 'foursquare'
MODEL_NAME = "STAMP"
device_id = '3'
epochs = 100
early_stopping = 5
embedding_size = 32
min_items_occurrences, min_users_interactions = 1, 5
topk_evalgrid = [5, 10, 25, 50, 100]

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
               'MAX_ITEM_LIST_LENGTH': 1000,
               'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
               'neg_sampling': None,
               'train_neg_sample_args': None,
               'epochs': epochs,
               'stopping_step': early_stopping,
               'embedding_size': embedding_size,
               'eval_args': {
                   'split': {'LS': 'valid_and_test'},
                   'group_by': 'user',
                   'order': 'TO',
                   'mode': 'pop100',
                   'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'ItemCoverage', 'AveragePopularity'],
                   'topk': [1, 5, 10],
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
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
if MODEL_NAME == 'GRU4Rec':
    model = GRU4Rec(config, train_data.dataset).to(config['device'])
elif MODEL_NAME == 'LightSANs':
    model = LightSANs(config, train_data.dataset).to(config['device'])
elif MODEL_NAME == 'STAMP':
    model = STAMP(config, train_data.dataset).to(config['device'])
else:
    raise Exception(f'{MODEL_NAME} is not supported.')
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# model evaluation
test_result = trainer.evaluate(test_data)
print(test_result)

# Compute bias metrics
# Coverage, HitRate by EI, HitRate by ItemPop
model.eval()
internal_user_ids = dataset.token2id('user_id', dataset.field2id_token['user_id'][1:])
internal_item_ids = dataset.token2id('item_id', dataset.field2id_token['item_id'][1:])
user_mapping_dict = {elem: i for i, elem in enumerate(internal_user_ids)}
item_mapping_dict = {elem: i for i, elem in enumerate(internal_item_ids)}

for topk in topk_evalgrid:
    hitrate_by_items, occurrences_by_items = np.zeros(len(internal_item_ids)), np.zeros(len(internal_item_ids))
    hitrate_by_users = np.zeros(len(internal_user_ids))
    all_predicted_items = []

    for _, _, batch_internal_user_ids, batch_internal_item_ids in test_data:
        _, topk_iid_list = full_sort_topk(batch_internal_user_ids, model, test_data, k=topk,
                                          device=config['device'])
        for i in range(len(batch_internal_user_ids) - 1):
            test_user = batch_internal_user_ids[i + 1].item()
            pred_items = topk_iid_list[i]
            ground_truth_item = batch_internal_item_ids[i].item()
            hitrate_by_items[item_mapping_dict[ground_truth_item]] += ground_truth_item in pred_items
            occurrences_by_items[item_mapping_dict[ground_truth_item]] += 1
            hitrate_by_users[user_mapping_dict[test_user]] += ground_truth_item in pred_items
            all_predicted_items += pred_items.detach().cpu().numpy().tolist()
            all_predicted_items = list(set(all_predicted_items))

    # Item stats
    coverage = len(set(all_predicted_items)) / len(internal_item_ids)
    norm_hitrate_by_items = hitrate_by_items / occurrences_by_items
    # Compute popularity of each item
    item_pop = np.full(fill_value=0, shape=hitrate_by_items.shape)
    for item_id in dataset.item_counter:
        item_pop[item_mapping_dict[item_id]] = dataset.item_counter[item_id]

    first_quartile_val = np.percentile(item_pop, 25)
    first_quartile_items = np.where(item_pop < first_quartile_val)[0]
    second_quartile_val = np.percentile(item_pop, 50)
    second_quartile_items = np.where((item_pop >= first_quartile_val) & (item_pop < second_quartile_val))[0]
    third_quartile_val = np.percentile(item_pop, 75)
    third_quartile_items = np.where((item_pop >= second_quartile_val) & (item_pop < third_quartile_val))[0]
    fourth_quartile_items = np.where(item_pop >= third_quartile_val)[0]
    avg_hitrate_by_pop = [np.nanmean(norm_hitrate_by_items[first_quartile_items]),
                          np.nanmean(norm_hitrate_by_items[second_quartile_items]),
                          np.nanmean(norm_hitrate_by_items[third_quartile_items]),
                          np.nanmean(norm_hitrate_by_items[fourth_quartile_items])]
    std_hitrate_by_pop = [np.nanstd(norm_hitrate_by_items[first_quartile_items]),
                          np.nanstd(norm_hitrate_by_items[second_quartile_items]),
                          np.nanstd(norm_hitrate_by_items[third_quartile_items]),
                          np.nanstd(norm_hitrate_by_items[fourth_quartile_items])]
    # User Stats
    # Get explorer-returner index
    explorer_index_df = pd.read_csv(base_dir / 'data' / 'processed' / DATASET_NAME / f'{DATASET_NAME}.user', sep='\t')
    explorer_index_vec = np.full(fill_value=None, shape=hitrate_by_users.shape)
    for internal_user_id in internal_user_ids:
        external_user_id = dataset.id2token('user_id', internal_user_id)
        # Get explorer index for this user
        ei_val = explorer_index_df[explorer_index_df['user_id:token'] == int(external_user_id)]['explorer_index:float'].item()
        explorer_index_vec[user_mapping_dict[internal_user_id]] = ei_val

    first_quartile_val = np.percentile(explorer_index_vec, 25)
    first_quartile_users = np.where(explorer_index_vec <= first_quartile_val)[0]
    second_quartile_val = np.percentile(explorer_index_vec, 50)
    second_quartile_users = np.where((explorer_index_vec > first_quartile_val) & (explorer_index_vec <= second_quartile_val))[0]
    third_quartile_val = np.percentile(explorer_index_vec, 75)
    third_quartile_users = np.where((explorer_index_vec > second_quartile_val) & (explorer_index_vec <= third_quartile_val))[0]
    fourth_quartile_users = np.where(explorer_index_vec > third_quartile_val)[0]
    avg_hitrate_by_ei = [hitrate_by_users[first_quartile_users].mean(),
                          hitrate_by_users[second_quartile_users].mean(),
                          hitrate_by_users[third_quartile_users].mean(),
                          hitrate_by_users[fourth_quartile_users].mean()]
    std_hitrate_by_ei = [hitrate_by_users[first_quartile_users].std(),
                          hitrate_by_users[second_quartile_users].std(),
                          hitrate_by_users[third_quartile_users].std(),
                          hitrate_by_users[fourth_quartile_users].std()]
    # Save results
    with open(output_data_dir / f'hitrate_by_itempop{topk}.pkl', 'wb') as file:
        pickle.dump({'avg': avg_hitrate_by_pop, 'std': std_hitrate_by_pop}, file)
    with open(output_data_dir / f'hitrate_by_userEI{topk}.pkl', 'wb') as file:
        pickle.dump({'avg': avg_hitrate_by_ei, 'std': std_hitrate_by_ei}, file)
    with open(output_data_dir / f'coverage{topk}.pkl', 'wb') as file:
        pickle.dump(coverage, file)
    with open(output_data_dir / f'hitrate{topk}.pkl', 'wb') as file:
        pickle.dump({'avg': np.nanmean(norm_hitrate_by_items), 'std': np.nanstd(norm_hitrate_by_items)}, file)
    print(f'--- TEST METRICS k={topk} ---')
    print(f'HitRate@{topk}: {round(np.nanmean(norm_hitrate_by_items),4)}')
    print(f'Coverage@{topk}: {round(coverage, 4)}')
    print(f'HitRateByPop@{topk}: {[round(elem, 2) for elem in avg_hitrate_by_pop]}')
    print(f'HitRateByEI@{topk}: {[round(elem, 2) for elem in avg_hitrate_by_ei]}')
