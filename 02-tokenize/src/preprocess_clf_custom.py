from transformers import TrainingArguments
from transformers import BertTokenizerFast
from transformers import BertConfig
from transformers import pipeline
from transformers import Trainer
from datasets import load_dataset
from datasets import DatasetDict
import transformers 
import pandas as pd
import numpy as np
import datasets
import logging
import pickle
import sys
import os


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Log versions of dependencies
logger.info(f'[Using Transformers: {transformers.__version__}]')
logger.info(f'[Using Datasets: {datasets.__version__}]')
logger.info(f'[Using Pandas: {pd.__version__}]')

# Essentials
# LOCAL_INPUT_PATH is mapped to S3 input location for covid article headlines 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# LOCAL_OUTPUT_PATH is mapped to S3 output location where we want to save the processed input data (COVID article headlines)
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'
MAX_LENGTH = 512
N_GPUS = 1

# Initiatize BERT custom tokenizer
config = BertConfig()
tokenizer = BertTokenizerFast.from_pretrained(f'{LOCAL_INPUT_PATH}/vocab', config=config)
tokenizer.model_max_length = MAX_LENGTH
tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
logger.info(f'Tokenizer: {tokenizer}')

# Load classification data (headlines of COVID new articles)
data = load_dataset('csv', 
                     data_files=f'{LOCAL_INPUT_PATH}/data/covid_articles_clf_data.csv', 
                     column_names=['text', 'label'], 
                     delimiter=',', 
                     split='train', 
                     cache_dir='/tmp/cache')
logger.info(f'Loaded data: {data}')


# Create data splits
train_validation_test = data.train_test_split(shuffle=True, seed=123, test_size=0.1)
validation_test = train_validation_test['test'].train_test_split(shuffle=True, seed=123, test_size=0.5)
data_splits = DatasetDict({'train': train_validation_test['train'],  
                           'validation': validation_test['train'], 
                           'test': validation_test['test']})
logger.info(f'Data splits: {data_splits}')


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

# Tokenize datasets 
num_proc = int(os.cpu_count()/N_GPUS)
logger.info(f'Total number of processes = {num_proc}')
tokenized_data = data_splits.map(preprocess_function, batched=True, num_proc=num_proc)
logger.info(f'Tokenized data: {tokenized_data}')


# Save tokenized data splits locally to the EBS volume attached to the Processing cluster
logger.info(f'Saving tokenized datasets to local disk {LOCAL_OUTPUT_PATH}')
tokenized_data.save_to_disk(f'{LOCAL_OUTPUT_PATH}')

# Validate if datasets were saved correctly
logger.info('Validating if datasets were saved correctly')
reloaded_datasets = datasets.load_from_disk(f'{LOCAL_OUTPUT_PATH}')
logger.info(f'Reloaded datasets: {reloaded_datasets}')