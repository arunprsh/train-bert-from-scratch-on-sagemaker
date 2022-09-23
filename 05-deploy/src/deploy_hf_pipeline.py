
import subprocess
import sys

from sagemaker.huggingface import HuggingFaceModel
import sagemaker
import argparse
import logging
import boto3

# Setup logging 


if __name__ == '__main__':
    # Parse argument variables passed via the DeployModel processing step
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--region', type=str)
    parser.add_argument('--deployment_instance_type', type=str)
    parser.add_argument('--deployment_instance_count', type=int)
    parser.add_argument('--model_s3_path', type=str)
    parser.add_argument('--endpoint_name', type=str)
    args, _ = parser.parse_known_args()

    
    region = args.region
    boto3.setup_default_session(region_name=region)
    role = sagemaker.get_execution_role()


    # Create HuggingFaceModel
    huggingface_model = HuggingFaceModel(model_data=f'{args.model_s3_path}/model.tar.gz',  # path to your trained sagemaker model 
                                         role=role, # iam role with permissions to create an Endpoint 
                                         transformers_version='4.6', # transformers version used 
                                         tensorflow_version='2.4', # tensorflow version used 
                                         py_version='py37', # python version of the DLC
                                        )
    
    logger.info('Deploying model ...')
    huggingface_model.deploy(initial_instance_count=args.deployment_instance_count, 
                             instance_type=args.deployment_instance_type)
    logger.info('Model deployment complete!')