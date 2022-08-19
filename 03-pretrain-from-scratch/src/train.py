from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import pipeline 
from datasets import load_dataset
from transformers import Trainer
from datasets import DatasetDict
import transformers
import sagemaker
import datasets
import argparse
import logging
import random
import shutil
import torch
import boto3
import time
import math
import sys
import os

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Log versions of dependencies
logger.info(f'[Using Transformers: {transformers.__version__}]')
logger.info(f'[Using SageMaker: {sagemaker.__version__}]')
logger.info(f'[Using Datasets: {datasets.__version__}]')
logger.info(f'[Using Torch: {torch.__version__}]')

# Essentials 
config = BertConfig()
s3 = boto3.resource('s3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logger.info('Parsing command line arguments')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training) 
    # are passed as command-line arguments to the training script
    parser.add_argument('--s3_bucket', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--per_device_train_batch_size', type=int)
    args, _ = parser.parse_known_args()
    
    CURRENT_HOST = args.current_host
    logger.info(f'Current host = {CURRENT_HOST}')
    num_gpus = args.num_gpus
    logger.info(f'Total number of GPUs per node = {num_gpus}')
    
    S3_BUCKET = args.s3_bucket
    MAX_LENGTH = args.max_len
    CHUNK_SIZE = args.chunk_size
    TRAIN_EPOCHS = args.num_train_epochs
    BATCH_SIZE = args.per_device_train_batch_size
    SAVE_STEPS = 10000
    SAVE_TOTAL_LIMIT = 2
    
    # Download saved custom vocabulary file from S3 to local input path of the training cluster
    logger.info(f'Downloading custom vocabulary from [{S3_BUCKET}/vocab/] to [{args.input_dir}/vocab/]')
    bucket = s3.Bucket(S3_BUCKET)
    path = os.path.join(f'{args.input_dir}', 'vocab')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(f'{path}/vocab.txt', 'wb') as data:
        bucket.download_fileobj('vocab/vocab.txt', data)
    
     
    # Re-create BERT WordPiece tokenizer 
    logger.info(f'Re-creating BERT tokenizer using custom vocabulary from [{args.input_dir}/vocab/]')
    tokenizer = BertTokenizerFast.from_pretrained(f'{args.input_dir}/vocab/', config=config)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
    logger.info(f'Tokenizer: {tokenizer}')

    # Read dataset and collate to create the mini batches for Masked Language Model (MLM) training
    logger.info('Reading and collating input data to create mini batches for Masked Language Model (MLM) training')
    # dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=f'{args.train}/covid_articles.txt', block_size=128)
    dataset = load_dataset('text', data_files=f'{args.train}/covid_articles.txt', split='train', cache_dir='/tmp/cache')
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
    num_proc = int(os.cpu_count()/num_gpus)
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
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm=True, 
                                                    mlm_probability=0.15)
        
    # Load MLM
    logger.info('Loading BertForMaskedLM model')
    mlm = BertForMaskedLM(config=config)
    
    # Train MLM
    logger.info('Training MLM')
    training_args = TrainingArguments(output_dir='/tmp/checkpoints', 
                                      overwrite_output_dir=True, 
                                      optim='adamw_torch',
                                      num_train_epochs=TRAIN_EPOCHS,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      evaluation_strategy='epoch',
                                      save_steps=SAVE_STEPS, 
                                      save_total_limit=SAVE_TOTAL_LIMIT)
    trainer = Trainer(model=mlm, 
                      args=training_args, 
                      data_collator=data_collator,
                      train_dataset=chunked_datasets['train'],
                      eval_dataset=chunked_datasets['validation'])
    trainer.train()
    
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    
    if CURRENT_HOST == 'algo-1':
        # Save trained model to local model directory
        logger.info(f'Saving trained MLM to [{args.model_dir}/custom/]')
        trainer.save_model(f'{args.model_dir}/custom')
        time.sleep(120)  # Wait for a few minutes to ensure the model is saved locally
    
        # Copy trained model from local directory of the training cluster to S3 
        logger.info(f'Copying saved model from local to [{S3_BUCKET}/model/custom/]')
        s3.meta.client.upload_file(f'{args.model_dir}/custom/pytorch_model.bin', S3_BUCKET, 'model/custom/pytorch_model.bin')
        s3.meta.client.upload_file(f'{args.model_dir}/custom/config.json', S3_BUCKET, 'model/custom/config.json')

        # [IMPORTANT] Copy vocab.txt to local model directory - this is needed to re-create the trained MLM
        logger.info('Copying custom vocabulary to local model artifacts location to faciliate model evaluation')
        shutil.copyfile(f'{args.input_dir}/vocab/vocab.txt', f'{args.model_dir}/custom/vocab.txt')
        
        # [IMPORTANT] Copy vocab.txt to saved model artifacts location in S3
        logger.info(f'Copying custom vocabulary from [{S3_BUCKET}/data/vocab/] to [{S3_BUCKET}/model/custom/] for future stages of ML pipeline')
        # TODO
        
        # Evaluate the trained model 
        logger.info('Create fill-mask task pipeline to evaluate trained MLM')
        fill_mask = pipeline('fill-mask', model=f'{args.model_dir}/custom')
        
        prediction = fill_mask('covid is a [MASK]')
        logger.info(prediction) 

        prediction = fill_mask('Delta [MASK] is a [MASK]')
        logger.info(prediction)

        prediction = fill_mask('Omicron [MASK] in US')
        logger.info(prediction)  