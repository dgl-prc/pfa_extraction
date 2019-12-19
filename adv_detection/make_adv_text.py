import torch
from data_factory.imdb_sentiment.imdb_data_process import *
from utils.save_function import load_pickle
from utils.constant import *

def main():

    dataProcessor = IMDB_Data_Processor(word2vec_model, stop_words_list_path)
    train_data = dataProcessor.load_data(train_data_path)
    model = load_pickle(ModelPath.IMDB.LSTM)


