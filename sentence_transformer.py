from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

class SentenceTransformer:
    def __init__(self, corpus: list):
        self.corpus = corpus

    @staticmethod
    def create_embedding(texts: list):
        tokens = {'input_ids': [], 'attention_mask': []}
        for sentence in texts:
            # encode each sentence and append to dictionary
            new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)

        # calculate the mean as the sum of the embedding activations summed divided by the number of values that should be given attention in each position summed_mask
        mean_pooled = summed / summed_mask

        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()
        return mean_pooled

    @staticmethod
    def calculate_similarity(corpus_embedding, query_embedding):
        return list(cosine_similarity(query_embedding, [corpus_embedding]))[0]

    def process(self, query: str):
        corpus_embedding = self.create_embedding(self.corpus)
        query_embedding = self.create_embedding([query])
        return {i: list(self.calculate_similarity(corpus_embedding[i], query_embedding))[0] for i in range(len(corpus_embedding))}