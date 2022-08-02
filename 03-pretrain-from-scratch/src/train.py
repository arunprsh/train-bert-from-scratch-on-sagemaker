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
logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# Log versions of dependencies
logger.info(f'[Using Transformers: {transformers.__version__}]')
logger.info(f'[Using SageMaker: {sagemaker.__version__}]')
logger.info(f'[Using Datasets: {datasets.__version__}]')
logger.info(f'[Using Torch: {torch.__version__}]')

# TODO
config = BertConfig()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training) are passed as command-line arguments to the training script
    logger.info('Handling command line arguments')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n_gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    args, _ = parser.parse_known_args()
    
    
    CURRENT_HOST = args.current_host
    logger.info(f'Current host = {CURRENT_HOST}')
    
    # Download saved custom vocabulary file from S3 to local input path of the training cluster
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('sagemaker-us-east-1-119174016168')
    path = os.path.join(f'{args.input_dir}', 'vocab')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(f'{path}/vocab.txt', 'wb') as data:
        bucket.download_fileobj('vocab/vocab.txt', data)
    
     

    logger.info('Re-creating BERT tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(f'{args.input_dir}/vocab/', config=config)
    tokenizer.model_max_length = 256
    tokenizer.init_kwargs['model_max_length'] = 256

    # Read dataset and collate to create the mini batches for MLM training
    dataset = LineByLineTextDataset(tokenizer=tokenizer, 
                                    file_path=f'{args.training_dir}/articles.txt', 
                                    block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    mlm = BertForMaskedLM(config=config)
    
    # Train Masked Language Model (MLM)
    training_args = TrainingArguments(output_dir="./covidBERT", 
                                      overwrite_output_dir=True, 
                                      num_train_epochs=40, 
                                      per_device_train_batch_size=32, 
                                      save_steps=10_000, 
                                      save_total_limit=2)
    trainer = Trainer(model=mlm, 
                      args=training_args, 
                      data_collator=data_collator, 
                      train_dataset=dataset)
    trainer.train()
    
    if CURRENT_HOST == 'algo-1':
    
        # Save trained model to local model directory
        trainer.save_model(f'{args.model_dir}/bert/')
        time.sleep(120)
        print(os.listdir(f'{args.model_dir}/bert/'))

        # Copy trained model from local directory of the training cluster to S3 
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(f'{args.model_dir}/bert/pytorch_model.bin', 'sagemaker-us-east-1-119174016168', 'model/' + 'pytorch_model.bin')
        s3.meta.client.upload_file(f'{args.model_dir}/bert/config.json', 'sagemaker-us-east-1-119174016168', 'model/' + 'config.json')

        # [IMPORTANT] Copy vocab.txt to local model directory - this is needed to re-create the trained model
        shutil.copyfile(f'{args.input_dir}/vocab/vocab.txt', f'{args.model_dir}/bert/vocab.txt')
    
    
        # Evaluate the trained model 
        logger.info('Create fill-mask task pipeline to evaluate trained MLM')
        fill_mask = pipeline('fill-mask', model=f'{args.model_dir}/bert/')
        prediction = fill_mask('covid is a [MASK]')
        logger.info(prediction) 

        prediction = fill_mask('Covid-19 is a [MASK]')
        logger.info(prediction)

        prediction = fill_mask('covid-19 is a [MASK]')
        logger.info(prediction)

        prediction = fill_mask('Covid is a [MASK]')
        logger.info(prediction)

        prediction = fill_mask('Omicron [MASK] in US')
        logger.info(prediction)  