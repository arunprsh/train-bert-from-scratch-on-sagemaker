from transformers import BertTokenizerFast
from transformers import BertConfig
from datasets import load_dataset
from datasets import DatasetDict
from pathlib import Path
import transformers 
import datasets
import logging
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

# Essentials
# LOCAL_INPUT_PATH is mapped to S3 input location for covid news articles 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# LOCAL_OUTPUT_PATH is mapped to S3 output location where we want to save the processed input data (COVID articles)
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'
MAX_LENGTH = 512
CHUNK_SIZE = 128
N_GPUS = 1

# Re-create BERT WordPiece tokenizer using the saved custom vocabulary from the previous job
config = BertConfig()
logger.info(f'Re-creating BERT tokenizer using custom vocabulary from [{LOCAL_INPUT_PATH}/vocab/]')
tokenizer = BertTokenizerFast.from_pretrained(f'{LOCAL_INPUT_PATH}/vocab', config=config)
tokenizer.model_max_length = MAX_LENGTH
tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
logger.info(f'Tokenizer: {tokenizer}')

# Read dataset and collate to create mini batches for Masked Language Model (MLM) training
logger.info('Reading and collating input data to create mini batches for Masked Language Model (MLM) training')
dataset = load_dataset('text', data_files=f'{LOCAL_INPUT_PATH}/data/covid_articles.txt', split='train', cache_dir='/tmp/cache')
logger.info(f'Dataset: {dataset}')

# Split dataset into train and validation splits 
logger.info('Splitting dataset into train and validation splits')
train_test_splits = dataset.train_test_split(shuffle=True, seed=123, test_size=0.1)
data_splits = DatasetDict({'train': train_test_splits['train'], 
                           'validation': train_test_splits['test']})
logger.info(f'Data splits: {data_splits}')
    

# Tokenize dataset
def tokenize(article):
    tokenized_article = tokenizer(article['text'])
    if tokenizer.is_fast:
        tokenized_article['word_ids'] = [tokenized_article.word_ids(i) for i in range(len(tokenized_article['input_ids']))]
    return tokenized_article


logger.info('Tokenizing dataset splits')
num_proc = int(os.cpu_count()/N_GPUS)
logger.info(f'Total number of processes = {num_proc}')
tokenized_datasets = data_splits.map(tokenize, batched=True, num_proc=num_proc, remove_columns=['text'])
logger.info(f'Tokenized datasets: {tokenized_datasets}')


# Concat and chunk dataset 
def concat_and_chunk(articles):
    # Concatenate all texts
    concatenated_examples = {key: sum(articles[key], []) for key in articles.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(articles.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length//CHUNK_SIZE) * CHUNK_SIZE
    # Split by chunks of max_len
    chunked_articles = {key: [text[i : i+CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)] for key, text in concatenated_examples.items()}
    # Create a new labels column
    chunked_articles['labels'] = chunked_articles['input_ids'].copy()
    return chunked_articles
    
logger.info('Concatenating and chunking the datasets to a fixed length')
chunked_datasets = tokenized_datasets.map(concat_and_chunk, batched=True, num_proc=num_proc)
logger.info(f'Chunked datasets: {chunked_datasets}')

# Save chunked datasets to local disk (EBS volume)
logger.info(f'Saving chunked datasets to local disk {LOCAL_OUTPUT_PATH}')
chunked_datasets.save_to_disk(f'{LOCAL_OUTPUT_PATH}')

# Validate if datasets were saved correctly
logger.info('Validating if datasets were saved correctly')
reloaded_dataset = datasets.load_from_disk(f'{LOCAL_OUTPUT_PATH}')
logger.info(f'Reloaded dataset: {reloaded_dataset}')