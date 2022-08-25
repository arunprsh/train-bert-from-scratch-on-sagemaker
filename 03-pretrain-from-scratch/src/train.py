from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import BertConfig
from transformers import pipeline 
from datasets import load_dataset
from transformers import Trainer
from datasets import DatasetDict
from pathlib import Path
import pandas as pd
import transformers
import sagemaker
import datasets
import argparse
import logging
import random
import shutil
import torch
import boto3
import json
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
logger.info(f'[Using Pandas: {pd.__version__}]')

# Essentials 
config = BertConfig()
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logger.info('Parsing command line arguments')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--master_host', type=str, default=os.environ['SMDATAPARALLEL_SERVER_ADDR'])
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training) 
    # are passed as command-line arguments to the training script
    parser.add_argument('--s3_bucket', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--per_device_train_batch_size', type=int)
    
    args, _ = parser.parse_known_args()
    current_host = args.current_host
    master_host = args.master_host
    
    logger.info(f'Current host = {current_host}')
    logger.info(f'Master host = {master_host}')
    
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
        
    # Copy preprocessed datasets from S3 to local EBS volume (cache dir)
    logger.info(f'Downloading preprocessed datasets from [{S3_BUCKET}/data/processed/] to [/tmp/cache/data/processed/]')
    def get_bucket_content(bucket, prefix=''):
        files = []
        folders = []
        default_kwargs = {'Bucket': bucket, 'Prefix': prefix}
        next_token = ''
        while next_token is not None:
            updated_kwargs = default_kwargs.copy()
            if next_token != '':
                updated_kwargs['ContinuationToken'] = next_token
            response = s3_client.list_objects_v2(**default_kwargs)
            contents = response.get('Contents')
            for result in contents:
                key = result.get('Key')
                if key[-1] == '/':
                    folders.append(key)
                else:
                    files.append(key)
            next_token = response.get('NextContinuationToken')
        return files, folders
    
    files, folders = get_bucket_content(S3_BUCKET, 'data/processed/')
    
    
    def copy_to_local_from_s3(bucket: str, local_path: str, files: list, folders: list) -> None:
        local_path = Path(local_path)
        for folder in folders:
            folder_path = Path.joinpath(local_path, folder)
            folder_path.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            file_path = Path.joinpath(local_path, file_name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, file_name, str(file_path))


    copy_to_local_from_s3(S3_BUCKET, '/tmp/cache', files, folders)
    
    
    # Re-create BERT WordPiece tokenizer 
    logger.info(f'Re-creating BERT tokenizer using custom vocabulary from [{args.input_dir}/vocab/]')
    tokenizer = BertTokenizerFast.from_pretrained(f'{args.input_dir}/vocab/', config=config)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
    logger.info(f'Tokenizer: {tokenizer}')

    # Read dataset 
    chunked_datasets = datasets.load_from_disk('/tmp/cache/data/processed')
    logger.info(f'Chunked datasets: {chunked_datasets}')
    
   
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
    
    
    if current_host == master_host:
        if not os.path.exists('/tmp/cache/model/custom'):
            os.makedirs('/tmp/cache/model/custom', exist_ok=True)

        # Save trained model to local model directory
        logger.info(f'Saving trained MLM to [/tmp/cache/model/custom/]')
        trainer.save_model('/tmp/cache/model/custom')
        
        logger.info(os.listdir('/tmp/cache/model/custom'))
        if os.path.exists('/tmp/cache/model/custom/pytorch_model.bin') and os.path.exists('/tmp/cache/model/custom/config.json'):
            # Copy trained model from local directory of the training cluster to S3 
            logger.info(f'Copying saved model from local to [s3://{S3_BUCKET}/model/custom/]')
            s3.meta.client.upload_file('/tmp/cache/model/custom/pytorch_model.bin', S3_BUCKET, 'model/custom/pytorch_model.bin')
            s3.meta.client.upload_file('/tmp/cache/model/custom/config.json', S3_BUCKET, 'model/custom/config.json')

            # Copy vocab.txt to local model directory - this is needed to re-create the trained MLM
            logger.info('Copying custom vocabulary to local model artifacts location to faciliate model evaluation')
            shutil.copyfile(f'{args.input_dir}/vocab/vocab.txt', '/tmp/cache/model/custom/vocab.txt')

            # Copy vocab.txt to saved model artifacts location in S3
            logger.info(f'Copying custom vocabulary from [{path}/vocab.txt] to [s3://{S3_BUCKET}/model/custom/] for future stages of ML pipeline')
            s3.meta.client.upload_file(f'{path}/vocab.txt', S3_BUCKET, 'model/custom/vocab.txt')

            # Evaluate the trained model 
            logger.info('Create fill-mask task pipeline to evaluate trained MLM')
            fill_mask = pipeline('fill-mask', model='/tmp/cache/model/custom')
            df = pd.read_csv(f's3://{S3_BUCKET}/data/eval/eval_mlm.csv')

            for gt, masked_sentence in zip(df.ground_truth.tolist(), df.masked.tolist()):
                logger.info(f'Ground Truth    : {gt}')
                logger.info(f'Masked sentence : {masked_sentence}')
                predictions = fill_mask(masked_sentence, top_k=3)
                for i, prediction in enumerate(predictions):
                    logger.info(f'Rank: {i+1} | {(prediction["score"] * 100):.2f} % | {[prediction["token_str"]]}')
            logger.info('-' * 10)