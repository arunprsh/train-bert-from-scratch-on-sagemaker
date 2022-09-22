from sklearn.metrics import precision_recall_fscore_support
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments
from transformers import BertTokenizerFast
from sagemaker.session import Session
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
import tarfile
import pickle
import boto3
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
    
    LOCAL_DATA_DIR = '/tmp/cache/data/bert/processed-clf'
    LOCAL_MODEL_DIR = '/tmp/cache/model/finetuned-clf'
    LOCAL_LABEL_DIR = '/tmp/cache/labels'
    
    # Setup SageMaker Session for S3Downloader and S3Uploader 
    boto_session = boto3.session.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    
    # Load BERT Sequence Model 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5,  force_download=True)
    
    # Re-create original BERT WordPiece tokenizer 
    logger.info('Re-creating original BERT tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    logger.info(f'Tokenizer: {tokenizer}')
    
    def download(s3_path: str, ebs_path: str, session: Session) -> None:
        try:
            if not os.path.exists(ebs_path):
                os.makedirs(ebs_path, exist_ok=True)
            S3Downloader.download(s3_path, ebs_path, sagemaker_session=session)
        except FileExistsError:  # to avoid race condition between GPUs
            logger.info('Ignoring FileExistsError to avoid I/O race conditions.')
        except FileNotFoundError:
            logger.info('Ignoring FileNotFoundError to avoid I/O race conditions.')
        
        
    def upload(ebs_path: str, s3_path: str, session: Session) -> None:
        S3Uploader.upload(ebs_path, s3_path, sagemaker_session=session)
    
    # Download preprocessed datasets from S3 to local EBS volume (cache dir)
    logger.info(f'Downloading preprocessed datasets from [{S3_BUCKET}/data/processed/] to [{LOCAL_DATA_DIR}/]')
    download(f's3://{S3_BUCKET}/data/bert/processed-clf/', f'{LOCAL_DATA_DIR}/', sm_session)
    
    # Load tokenized dataset 
    tokenized_data = datasets.load_from_disk(f'{LOCAL_DATA_DIR}/')
    logger.info(f'Tokenized data: {tokenized_data}')
    
    # Define compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1_macro': f1, 'precision': precision, 'recall': recall}
    
    # Fine-tune 
    training_args = TrainingArguments(output_dir='/tmp/checkpoints', 
                                      overwrite_output_dir=True, 
                                      optim='adamw_torch', 
                                      per_device_train_batch_size=BATCH_SIZE, 
                                      per_device_eval_batch_size=BATCH_SIZE, 
                                      evaluation_strategy='epoch',
                                      num_train_epochs=TRAIN_EPOCHS,  
                                      save_steps=SAVE_STEPS,
                                      save_total_limit=SAVE_TOTAL_LIMIT)
    
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=tokenized_data['train'], 
                      eval_dataset=tokenized_data['validation'], 
                      tokenizer=tokenizer, 
                      compute_metrics=compute_metrics)
    
    # Evaluate 
    train_metrics = trainer.train()
    logger.info(f'Train metrics: {train_metrics}')
    
    # Evaluate validation set 
    val_metrics = trainer.evaluate()
    logger.info(f'Validation metrics: {val_metrics}')
    
    # Evaluate test set (holdout)
    test_metrics = trainer.evaluate(eval_dataset=tokenized_data['test'])
    logger.info(f'Holdout metrics: {test_metrics}')
    
    
    def get_file_paths(directory: str) -> list:
        file_paths = [] 
        for root, directories, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)  
        return file_paths
    
    
    def tar_artifacts(local_artifacts_path: str, tar_save_path: str, tar_name: str) -> None:
        if not os.path.exists(tar_save_path):
            os.makedirs(tar_save_path, exist_ok=True)
        tar = tarfile.open(f'{tar_save_path}/{tar_name}', 'w:gz')
        file_paths = get_file_paths(local_artifacts_path)  
        for file_path in file_paths:
            file_ = file_path.split('/')[-1]
            try:
                tar.add(file_path, arcname=file_) 
            except OSError:
                logger.info('Ignoring OSErrors during tar creation.')
        tar.close()
    
    if current_host == master_host:
        if not os.path.exists(LOCAL_MODEL_DIR):
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        if not os.path.exists(LOCAL_LABEL_DIR):
            os.makedirs(LOCAL_LABEL_DIR, exist_ok=True)
            
        # Download label mapping from s3 to local
        download(f's3://{S3_BUCKET}/data/labels/', f'{LOCAL_LABEL_DIR}/', sm_session)
    
        # Load label mapping for inference
        with open(f'{LOCAL_LABEL_DIR}/label_map.pkl', 'rb') as f:
            label2id = pickle.load(f)
        
        id2label = dict((str(v), k) for k, v in label2id.items())
        logger.info(f'Label mapping: {id2label}')
    
        trainer.model.config.label2id = label2id
        trainer.model.config.id2label = id2label
        
        # Save finetuned model to local  
        trainer.save_model(LOCAL_MODEL_DIR)
        
        if os.path.exists(f'{LOCAL_MODEL_DIR}/pytorch_model.bin') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/training_args.bin') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/config.json') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/special_tokens_map.json') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/tokenizer_config.json') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/tokenizer.json') and \
            os.path.exists(f'{LOCAL_MODEL_DIR}/vocab.txt'):
            
            # Copy trained model from local directory of the training cluster to S3 
            logger.info(f'Copying saved model from local to [s3://{S3_BUCKET}/model/finetuned-clf/]')
            upload(LOCAL_MODEL_DIR, f's3://{S3_BUCKET}/model/finetuned-clf', sm_session) 
        
            # Test model for inference
            classifier = pipeline('sentiment-analysis', model=LOCAL_MODEL_DIR)
            prediction = classifier('Covid pandemic is still raging in may parts of the world')
            logger.info(prediction)
            
            # Save model as tar to local
            tar_artifacts(LOCAL_MODEL_DIR, f'{LOCAL_MODEL_DIR}/model-tar', 'model.tar.gz')
            
            # Upload model tar to S3 from local
            logger.info(f'Copying saved model tar from local to [s3://{S3_BUCKET}/model/finetuned-clf/model-tar/]')
            upload(f'{LOCAL_MODEL_DIR}/model-tar', f's3://{S3_BUCKET}/model/finetuned-clf/model-tar', sm_session) 
            
            # Save pipeline to local 
            classifier.save_pretrained(f'{LOCAL_MODEL_DIR}/pipeline')
            
            # Save pipeline as tar to local
            tar_artifacts(f'{LOCAL_MODEL_DIR}/pipeline', f'{LOCAL_MODEL_DIR}/pipeline-tar', 'pipeline.tar.gz')
            
            # Upload pipeline tar to S3 from local
            logger.info(f'Copying saved pipeline tar from local to [s3://{S3_BUCKET}/model/finetuned-clf/pipeline-tar/]')
            upload(f'{LOCAL_MODEL_DIR}/pipeline-tar', f's3://{S3_BUCKET}/model/finetuned-clf/pipeline-tar', sm_session)