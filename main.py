import numpy as np
import pandas as pd

from base_encoder import Encoder
from bert import BERT
from glove import GloVe
from sentence_transformer import SentenceTransformer
from universal_sentence_encoder import UniversalSentenceEncoder
from helper_functions import write_res_to_db

def content_based_train_infer(train_df: pd.DataFrame, test_df: pd.DataFrame):
    '''Calculates content based similarity scores between items in test_df and items in train_df as the candidates using 4 different methods; writes result and performance metrics to database'''
    encoder = Encoder(train_df=train_df, test_df=test_df)
    for col_name, method in zip(['BERT','GloVe','sentence_transformer','universal_sentence_encoder'], [BERT, GloVe, SentenceTransformer, UniversalSentenceEncoder]):
        print(f'Processing for {col_name}')
        encoder.test_df[f'{col_name}_recommendation'], encoder.test_df[f'{col_name}_recommended_items'] = encoder.get_result(method)
        encoder.test_df['personalization'] = encoder.test_df[f'{col_name}_recommended_items'].map(encoder.calculate_personalization)
        encoder.test_df['diversity'] = encoder.test_df[f'{col_name}_recommended_items'].map(encoder.calculate_diversity)
        encoder.test_df['coverage'] = encoder.test_df[f'{col_name}_recommended_items'].map(encoder.calculate_coverage)
        personalization, diversity, coverage = np.mean(encoder.test_df['personalization']), np.mean(encoder.test_df['diversity']), np.mean(encoder.test_df['coverage'])
        print(f'-----------Metrics for {col_name}-----------\nPersonalization: {personalization}\nDiversity: {diversity}\nCoverage: {coverage}')
        write_res_to_db(encoder.test_df, col_name)


if __name__ == '__main__':
    content_based_train_infer()
