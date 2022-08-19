import re
import numpy as np
from nltk.corpus import stopwords
stops = list(stopwords.words('english'))
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
tf.disable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)
similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)

class UniversalSentenceEncoder:
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.corpus_embeddings = self.create_embedding(self.corpus)

    def create_embedding(self, texts: list):
        corpus = [self.preprocess(text) for text in texts]
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: corpus})
        return embeddings_

    @staticmethod
    def calculate_similarity(corpus_embeddings_: np.ndarray, query_embeddings_: np.ndarray):
        corr = np.inner(corpus_embeddings_, query_embeddings_)
        return {i: corr[i][0] for i in range(len(corr))}

    def process(self, query: str):
        query_embeddings_ = self.create_embedding([query])
        return self.calculate_similarity(self.corpus_embeddings_, query_embeddings_)

    @staticmethod
    def preprocess(doc):
        # Tokenize, clean up input document string
        doc = re.sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = re.sub(r'<[^<>]+(>|$)', " ", doc)
        doc = re.sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        doc = re.sub(r'(%s)'%('|').join(stopwords),'',doc)
        return doc