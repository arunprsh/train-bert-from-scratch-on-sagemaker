from transformers import BertForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from transformers import TrainingArguments
from transformers import BertTokenizerFast
from datasets import load_dataset
from transformers import pipeline 
from transformers import Trainer
from datasets import DatasetDict
import pandas as pd
import transformers
import sagemaker
import argparse
import datasets
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
    # Hyperparameters sent by the client (Studio notebook with the driver code to launch training)
    # are passed as command-line arguments to the training script
    parser = argparse.ArgumentParser()
    logger.info('Handling command line arguments')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    args, _ = parser.parse_known_args()
    
    CURRENT_HOST = args.current_host
    
    print(args)
    print('-' * 20)
    print(os.listdir('/opt/ml'))
    print('-' * 20)
    
    logger.info(f'Current host = {CURRENT_HOST}')
    logger.info(f'Input directory = {args.input_dir}')
    logger.info(f'Model directory = {args.model_dir}')
    logger.info(f'Train directory = {args.train_dir}')
    
    # Download saved custom vocabulary file from S3 to local input path of the training cluster
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('sagemaker-us-east-1-119174016168')
    path = os.path.join(f'{args.input_dir}', 'vocab')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(f'{path}/vocab.txt', 'wb') as data:
        bucket.download_fileobj('vocab/vocab.txt', data)
    
    # Re-create BERT tokenizer
    logger.info('Re-creating BERT tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(f'{args.input_dir}/vocab/')
    tokenizer.model_max_length = 512
    tokenizer.init_kwargs['model_max_length'] = 512
    
    # Load and split dataset 
    data = datasets.load_dataset('csv', 
                                 data_files=f'{args.train_dir}/covid_articles_clf_data.csv', 
                                 column_names=['label', 'text'], 
                                 delimiter=',', 
                                 split='train', 
                                 cache_dir='/tmp/cache')
    train_dev_test = data.train_test_split(shuffle=True, seed=123, test_size=0.1)
    dev_test = train_dev_test['test'].train_test_split(shuffle=True, seed=123, test_size=0.5)
    data_splits = DatasetDict({'train': train_dev_test['train'], 
                               'test': dev_test['test'], 
                               'dev': dev_test['train']})
    
    # Tokenize data splits
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_data = data_splits.map(preprocess_function, batched=True)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    
    path = os.path.join(f'{args.input_dir}', 'bert')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(f'{path}/pytorch_model.bin', 'wb') as data:
        bucket.download_fileobj('model/pytorch_model.bin', data)
    
    """
    with open(f'{path}/training_args.bin', 'wb') as data:
        bucket.download_fileobj('model/training_args.bin', data)
    """
    
    with open(f'{path}/config.json', 'wb') as data:
        bucket.download_fileobj('model/config.json', data)
    
    
    
    
    
    # [IMPORTANT] Copy vocab.txt to local model directory - this is needed to re-create the trained model
    shutil.copyfile(f'{args.input_dir}/vocab/vocab.txt', f'{args.input_dir}/bert/vocab.txt')
    
    
    
    
    # Load BERT model
    model = BertForSequenceClassification.from_pretrained(f'{args.input_dir}/bert/', num_labels=5, force_download=True)
    
    # Set training args
    training_args = TrainingArguments(output_dir=f'tmp/checkpoints',
                                      learning_rate=2e-5, 
                                      per_device_train_batch_size=16, 
                                      per_device_eval_batch_size=16, 
                                      num_train_epochs=5,  
                                      weight_decay=0.01, 
                                      save_total_limit=2, 
                                      save_strategy='no', 
                                      load_best_model_at_end=False)
    # Fine-tune the model
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=tokenized_data['train'], 
                      eval_dataset=tokenized_data['test'], 
                      tokenizer=tokenizer, 
                      data_collator=data_collator)
    trainer.train()
    
    # Save fine-tuned model to local model directory
    trainer.save_model(f'{args.model_dir}/fine-tuned/')
    time.sleep(120)
    print(os.listdir(f'{args.model_dir}/fine-tuned/'))
    
    # Copy trained model from local directory of the training cluster to S3 
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(f'{args.model_dir}/fine-tuned/pytorch_model.bin', 
                               'sagemaker-us-east-1-119174016168', 
                               'model/fine-tuned-clf' + 'pytorch_model.bin')
    s3.meta.client.upload_file(f'{args.model_dir}/fine-tuned/config.json', 
                               'sagemaker-us-east-1-119174016168', 
                               'model/fine-tuned-clf' + 'config.json')
    
    # Load the model and create a classification pipeline 
    logger.info('Evaluate fine-tuned model')
    classifier = pipeline('sentiment-analysis', model=f'{args.model_dir}/fine-tuned/')
    result = classifier('I hate you')