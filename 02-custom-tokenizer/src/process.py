from tokenizers import BertWordPieceTokenizer
from pathlib import Path
import transformers 
import tokenizers
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
logger.info(f'[Using Tokenizers: {tokenizers.__version__}]')

# Essentials
# is mapped to S3 input location for covid articles 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# is mapped to S3 output location where we want to save the custom vocabulary from the trained tokenizer
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'
VOCAB_SIZE = 30522

# Read input files from local input path 
logger.info(f'Reading input files from {LOCAL_INPUT_PATH}')
paths = [str(x) for x in Path(LOCAL_INPUT_PATH).glob('*.txt')]

# Train custom BertWordPiece tokenizer
logger.info(f'Training BertWordPiece custom tokenizer using files in {paths}')
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=paths, vocab_size=VOCAB_SIZE)

# Save trained custom tokenizer to local output path
logger.info('Saving trained tokenizer to local output location')
tokenizer.save_model(LOCAL_OUTPUT_PATH)