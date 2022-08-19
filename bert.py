from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')

class BERT:
    def __init__(self, corpus: list):
        self.corpus = model.encode(corpus)

    def process(self, query: str):
        query_embedding = model.encode([query])
        return {i: list(list(cosine_similarity([self.corpus[i]], query_embedding))[0])[0] for i in range(len(self.corpus))}