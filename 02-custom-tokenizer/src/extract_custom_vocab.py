from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import transformers 
import pandas as pd
import logging
import os

logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

logging.info(f'[Using transformers: {transformers.__version__}]')

corpus_path = '/opt/ml/processing/input'

paths = [str(x) for x in Path(corpus_path).glob('*.txt')]
logger.info(f'Reading files in {paths}')

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=paths, vocab_size=30522)

tokenizer.save_model('/opt/ml/processing/output', prefix='tokenizer/')