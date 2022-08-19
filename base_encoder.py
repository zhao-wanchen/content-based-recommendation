from cgi import test
import sys
import os
import pandas as pd
import recmetrics
import itertools
import numpy as np
from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))
sys.path.append('rs-utils')

from config import n, column_keys
# n: maximum items to recommend

class Encoder:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df
        self.item_col = column_keys.item_col
        self.text_col = column_keys.text_col
        self.item_name_col = column_keys.page_title_col
        self.catalog = list(self.train_df[self.item_col].unique())

    def get_result(self, encoder):
        '''Creates text embedding and computes similarity scores between items in train_df and test_df. This can be applied to any of the techniques'''
        if (self.text_col not in self.train_df.columns or self.text_col not in self.test_df.columns):
            raise KeyError(f'{self.text_col} not in data, please verify')
        scores, items = [], []
        encoder = encoder(corpus = list(self.train_df[self.text_col]))
        items = list(self.train_df[self.item_name_col])
        for txt in self.test_df[self.text_col]:
            similarity_scores = encoder.process(txt)
            similarity_scores = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}
            similarity_scores = {k: similarity_scores[k] for k in list(similarity_scores.keys())[0:n]}
            scores.append(similarity_scores)
            items.append([items[i] for i in similarity_scores.keys()])
        return scores, items

    @staticmethod
    def calculate_personalization(recs: list):
        if len(recs) > 0:
            return recmetrics.personalization(predicted = recs)
        return 0

    @staticmethod
    def calculate_diversity(recs: list):
        all_items = list(set(itertools.chain.from_iterable(recs)))
        n = np.mean([len(rec) for rec in recs])
        return len(all_items) / len(recs) / n

    def calculate_coverage(self, recs: list):
        coverage = recmetrics.prediction_coverage(predicted=recs, catalog=self.catalog)
        return coverage




        