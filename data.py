import numpy as np
import datasets
import pandas as pd
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer


class ComplexityMeasurer:
    def __init__(self, corpus_sentences):
        # Load sentence transformer model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        corpus_embeddings = self.model.encode(corpus_sentences)
        self.corpus_mean_embedding = np.mean(corpus_embeddings, axis=0)
        self.min_score = 5.5
        self.max_score = 6.45

    def vector_entropy(self, vector):
        """Compute entropy of the vector."""
        normalized_vector = vector / np.linalg.norm(vector)
        return entropy(np.abs(normalized_vector))

    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def semantic_complexity(self, sentence):
        # Encode the sentence to get the embedding
        embedding = self.model.encode(sentence)
        # Measure vector entropy
        entropy_score = self.vector_entropy(embedding)
        # Measure deviation from corpus mean
        deviation_score = 1 - self.cosine_similarity(embedding, self.corpus_mean_embedding)
        # Combine the scores
        complexity_score = entropy_score + deviation_score
        return complexity_score

    def normalize_scores(self, score):
        """Normalize a list of scores to the range [0, 1]."""
        normalized_score = (score - self.min_score) / (self.max_score - self.min_score)
        return normalized_score

    def complexity(self, embedding):
        """Measure the complexity of the input."""
        entropy_score = self.vector_entropy(embedding)
        deviation_score = 1 - self.cosine_similarity(embedding, self.corpus_mean_embedding)
        complexity_score = entropy_score + deviation_score
        return self.normalize_scores(complexity_score)

class Dataset:
    def __init__(self, data, size):
        self.size = size
        self.X = data[:self.size]['text']
        self.vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.dim = self.vectorizer.encode(self.X[0]).shape[-1]
        self.cursor = 0
    
    def __len__(self):
        return self.size

    def step(self):
        assert self.cursor < self.size, "End of dataset"
        item = self.vectorizer.encode(self.X[self.cursor])
        self.cursor += 1
        return item

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0

data = datasets.Dataset.from_pandas(pd.read_csv("./imdb_preprocessed.csv"))
data.shuffle(seed=42)
num_sample = 1000
measurer = ComplexityMeasurer(data[:num_sample]['text'])

def get_dataset():
    return Dataset(data, num_sample)