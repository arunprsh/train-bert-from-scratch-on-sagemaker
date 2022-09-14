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
# LOCAL_INPUT_PATH is mapped to S3 input location for covid news articles 
LOCAL_INPUT_PATH = '/opt/ml/processing/input' 
# LOCAL_OUTPUT_PATH is mapped to S3 output location where we want to save the custom vocabulary after training the tokenizer
LOCAL_OUTPUT_PATH = '/opt/ml/processing/output'
VOCAB_SIZE = 30522

# Read input files from local input path 
logger.info(f'Reading input files from [{LOCAL_INPUT_PATH}/]')
paths = [str(x) for x in Path(LOCAL_INPUT_PATH).glob('*.txt')]

# Train custom BertWordPiece tokenizer
logger.info(f'Training BertWordPiece custom tokenizer using files in {paths}')
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=paths, vocab_size=VOCAB_SIZE)

# Save trained custom tokenizer to local output path
logger.info(f'Saving extracted custom vocabulary to [{LOCAL_OUTPUT_PATH}/]')
tokenizer.save_model(LOCAL_OUTPUT_PATH)

# Re-create custom tokenizer using vocab from local output path
logger.info(f'Re-create BertWordPiece custom tokenizer using extracted custom vocab in {LOCAL_OUTPUT_PATH}')
tokenizer = BertWordPieceTokenizer(f'{LOCAL_OUTPUT_PATH}/vocab.txt')

# Evaluate custom tokenizer 
logger.info('Evaluating custom tokenizer')
test_sentence = 'covid19 is a virus'
logger.info(f'Test sentence: {test_sentence}')
tokens = tokenizer.encode(test_sentence).tokens
logger.info(f'Encoded sentence: {tokens}')
token_id = tokenizer.token_to_id('covid19')
logger.info(f'Token ID for token (covid19) = {token_id}')
vocab_size = tokenizer.get_vocab_size()
logger.info(f'Vocabulary size = {vocab_size}')