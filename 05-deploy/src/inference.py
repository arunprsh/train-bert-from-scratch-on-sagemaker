from transformers import pipeline
import logging 


# Setup logging 


if __name__ == '__main__':
    classifier = pipeline('sentiment-analysis', model='./clf-model/checkpoint-1500')
    prediction = classifier('I love you')[0]