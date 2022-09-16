from sklearn.metrics import precision_recall_fscore_support
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import BertTokenizerFast
from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader
from datasets import load_dataset
from transformers import pipeline
from transformers import Trainer
from datasets import DatasetDict
import pandas as pd
import transformers
import numpy as np
import sagemaker
import argparse
import datasets
import sklearn
import logging 
import pickle
import torch
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
logger.info(f'[Using Sklearn: {sklearn.__version__}]')
logger.info(f'[Using Torch: {torch.__version__}]')
logger.info(f'[Using Pandas: {pd.__version__}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logger.info('Handling command line arguments')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--master_host', type=str, default=os.environ['SMDATAPARALLEL_SERVER_ADDR'])
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training) 
    # are passed as command-line arguments to the training script
    parser.add_argument('--s3_bucket', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--per_device_train_batch_size', type=int)
    parser.add_argument('--region', type=str)
    
    args, _ = parser.parse_known_args()
    current_host = args.current_host
    master_host = args.master_host
    
    logger.info(f'Current host = {current_host}')
    logger.info(f'Master host = {master_host}')
    
    S3_BUCKET = args.s3_bucket
    MAX_LENGTH = args.max_len
    TRAIN_EPOCHS = args.num_train_epochs
    BATCH_SIZE = args.per_device_train_batch_size
    REGION = args.region
    SAVE_STEPS = 10000
    SAVE_TOTAL_LIMIT = 2
    
    # Setup SageMaker Session for S3Downloader and S3Uploader 
    boto_session = boto3.session.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    
    # Load BERT Sequence Model 
    
    
    # Download saved custom vocabulary file from S3 to local input path of the training cluster
    logger.info(f'Downloading custom vocabulary from [{S3_BUCKET}/data/vocab/] to [{args.input_dir}/vocab/]')
    path = os.path.join(f'{args.input_dir}', 'vocab')
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    S3Downloader.download(f's3://{S3_BUCKET}/data/vocab/', f'{path}/', sagemaker_session=sm_session)
    
    
    # Re-create original BERT WordPiece tokenizer 
    logger.info('Re-creating original BERT tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(path, config=config)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
    logger.info(f'Tokenizer: {tokenizer}')
    
    path = '/tmp/cache/data/processed-clf/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # Download preprocessed datasets from S3 to local EBS volume (cache dir)
    logger.info(f'Downloading preprocessed datasets from [{S3_BUCKET}/data/processed/] to [{path}]')
    S3Downloader.download(f's3://{S3_BUCKET}/data/processed-clf/', path, sagemaker_session=sm_session)
    
    # Load tokenized dataset 
    tokenized_data = datasets.load_from_disk(path)
    logger.info(f'Tokenized data: {tokenized_data}')
    
    # Fine-tune 
    training_args = TrainingArguments(output_dir='./tmp', 
                                      overwrite_output_dir=True, 
                                      optim='adamw_torch', 
                                      learning_rate=2e-5, 
                                      per_device_train_batch_size=BATCH_SIZE, 
                                      per_device_eval_batch_size=BATCH_SIZE, 
                                      num_train_epochs=TRAIN_EPOCHS,  
                                      weight_decay=0.01, 
                                      save_total_limit=2, 
                                      save_strategy='no',  
                                      load_best_model_at_end=False)
    
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=tokenized_data['train'], 
                      eval_dataset=tokenized_data['validation'], 
                      tokenizer=tokenizer, 
                      compute_metrics=compute_metrics)
    
    # Evaluate 
    train_results = trainer.train()
    trainer.log_metrics('train', train_results.metrics)
    trainer.save_metrics('train', train_results.metrics)
    
    # Evaluate validation set results
    results = trainer.evaluate()
    trainer.log_metrics('validation', results)
    trainer.save_metrics('validation', results)
    
    # Evaluate test set results 
    results = trainer.evaluate(eval_dataset=tokenized_data['test'])
    trainer.log_metrics('test', results)
    trainer.save_metrics('test', results)
    
    # Download label mapping from s3 to local
    S3Downloader.download(f's3://{S3_BUCKET}/data/eval/', '/tmp/cache/eval/')
    
    # Load label mapping for inference
    with open('/tmp/cache/eval/label_map', 'rb') as f:
        label2id = pickle.load(f)
        
    id2label = dict((str(v), k) for k, v in label2id.items())
    logger.info(id2label)
    
    trainer.model.config.label2id = label2id
    trainer.model.config.id2label = id2label
    
    # Save model                     
    trainer.save_model('/tmp/cache/model/finetuned-clf/')

    if os.path.exists('/tmp/cache/model/finetuned-clf/pytorch_model.bin') and os.path.exists('/tmp/cache/model/finetuned-clf/config.json'):
        # Copy trained model from local directory of the training cluster to S3 
        logger.info(f'Copying saved model from local to [s3://{S3_BUCKET}/model/finetuned-clf/]')
        S3Uploader.upload('/tmp/cache/model/finetuned-clf/', f's3://{S3_BUCKET}/model/finetuned-clf/', sagemaker_session=sm_session)  
        
    # Test model for inference
    classifier = pipeline('sentiment-analysis', model=f'/tmp/cache/model/finetuned-clf/')
    prediction = classifier('I hate you')
    logger.info(prediction)