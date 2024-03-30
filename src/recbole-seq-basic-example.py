import pathlib

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, ModelType

DATASET_NAME = 'yelp-sample'

base_dir = pathlib.Path.cwd()
params_dict = {'train_neg_sample_args': None,
               'dataset': DATASET_NAME,
               'MODEL_TYPE': ModelType.SEQUENTIAL,
               'data_path': str(base_dir / 'data' / 'raw'),
               'TIME_FIELD': 'timestamp',
               'USER_ID_FIELD': 'user_id',
               'ITEM_ID_FIELD': 'item_id',
               'user_inter_num_interval': "[1,Inf)",
               'item_inter_num_interval': "[1,Inf)",
                'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
               }

# configurations initialization
config = Config(model='BERT4Rec', config_dict=params_dict)
config['epochs'] = 5
config['stopping_step'] = 2

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
model = BERT4Rec(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# model evaluation
test_result = trainer.evaluate(test_data)
test_result


model = 'GRU4Rec'
dataset = 'Yelp2022'

environment_dict_setting = {'gpu_id': '0',
                            'worker': 4,
                            'seed': 0,
                            'state': 'INFO',
                            'reproducibility': True,
                            'data_path': '../data/raw/',
                            'checkpoint_dir': '../data/interim/',
                            'show_progress': True,
                            'save_dataset': True,
                            }

data_dict_setting = {'ITEM_LIST_LENGTH_FIELD': 'item_length',
                     'LIST_SUFFIX': "_list",
                     'MAX_ITEM_LIST_LENGTH': 1000,
                     'POSITION_FIELD': 'position_id',
                     'rm_dup_inter': None,  # do not want to remove duplicated user-item entries
                     }

train_dict_setting = {'epochs': 10,
                      'train_batch_size': 1024,
                      'learner': 'adam',
                      'learning_rate': 1e-3,
                      'eval_step': 1,
                      }

eval_dict_setting = {'order': 'TO',
                     'split': {'LS': 'valid_and_test'},
                     'mode': 'uni100',
                     'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
                     'topk': [1, 5, 10],
                     'valid_metric': 'Hit@10'
                     }
MAX_ITEM = 20

parameter_dict = {
    'data_path': '/kaggle/working/',
    'USER_ID_FIELD': 'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    'user_inter_num_interval': "[5,Inf)",
    'item_inter_num_interval': "[5,Inf)",
    'load_col': {'inter': ['session', 'aid', 'ts']},
    'train_neg_sample_args': None,
    'epochs': 10,
    'stopping_step': 3,

    'eval_batch_size': 1024,
    # 'train_batch_size': 1024,
    #    'enable_amp':True,
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,
    'eval_args': {
        'split': {'RS': [9, 1, 0]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'}
}

# configurations initialization
config = Config(
    model=model,
    dataset=dataset,
    config_dict=parameter_dict,
)

init_seed(config["seed"], config["reproducibility"])
