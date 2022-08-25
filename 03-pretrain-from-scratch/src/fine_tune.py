from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, LineByLineTextDataset, BertTokenizerFast, BertForMaskedLM, BertConfig, pipeline 
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training)
    # are passed as command-line arguments to the training script
    logger.info('Handling command line arguments')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    args, _ = parser.parse_known_args()
    
    CURRENT_HOST = args.current_host
    logger.info(f'Current host = {CURRENT_HOST}')
    
    # Download default vocabulary file from S3 to local input path of the training cluster
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('sagemaker-us-east-1-119174016168')
    path = os.path.join(f'{args.input_dir}', 'vocab2')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(f'{path}/vocab.txt', 'wb') as data:
        bucket.download_fileobj('vocab2/vocab.txt', data)
    
    config = BertConfig()
    
    
    model = BertForMaskedLM(config=config)
    logger.info(f'Number of parameters = {model.num_parameters()}')
    
    tokenizer = BertTokenizerFast.from_pretrained(path, config=config)
    tokenizer.model_max_length = 512
    tokenizer.init_kwargs['model_max_length'] = 512

    
    
    

    
    