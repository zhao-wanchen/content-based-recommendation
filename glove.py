from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity

import re
from nltk.corpus import stopwords
stops = list(stopwords.words('english'))

glove = api.load("glove-wiki-gigaword-50")    
similarity_index = WordEmbeddingSimilarityIndex(glove)

class GloVe:
    def __init__(self, corpus: list):
        self.corpus = corpus

    @staticmethod
    def preprocess(doc):
        '''Tokenize, clean up input document string'''
        doc = re.sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = re.sub(r'<[^<>]+(>|$)', " ", doc)
        doc = re.sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        tokens = simple_preprocess(doc, min_len=0, max_len=float("inf"))
        return [token for token in tokens if token not in stopwords]

    def process(self, query: str):
        '''Conducts preprocessing, creates BoW corpus and computes text similarity scores'''
        corpus = [self.preprocess(text) for text in self.corpus]
        dictionary = Dictionary(corpus)
        tfidf = TfidfModel(dictionary=dictionary)

        # Create the term similarity matrix.  
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)
        query_tf = tfidf[dictionary.doc2bow(self.preprocess(query))]
        index = SoftCosineSimilarity(tfidf[[dictionary.doc2bow(document) for document in corpus]], similarity_matrix)
        scores = list(index[query_tf])
        return {i: scores[i] for i in range(len(scores))}