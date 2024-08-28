import numpy as np
import datasets
import pandas as pd
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer


class ComplexityMeasurer:
    def __init__(self, corpus_sentences):
        # Load the sentence transformer model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        corpus_embeddings = self.model.encode(corpus_sentences)
        self.corpus_mean_embedding = np.mean(corpus_embeddings, axis=0)
        
        # Compute initial complexity scores to establish min and max scores dynamically
        initial_scores = [self.semantic_complexity(self.model.encode(sentence)) for sentence in corpus_sentences]
        self.min_score = min(initial_scores)
        self.max_score = max(initial_scores)

    def vector_entropy(self, vector):
        """Compute entropy of the vector."""
        normalized_vector = vector / np.linalg.norm(vector)
        return entropy(np.abs(normalized_vector))

    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def semantic_complexity(self, embedding):
        """Calculate the semantic complexity of a sentence."""
        entropy_score = self.vector_entropy(embedding)
        deviation_score = 1 - self.cosine_similarity(embedding, self.corpus_mean_embedding)
        complexity_score = 0.5 * entropy_score + 0.5 * deviation_score
        return complexity_score

    def normalize_scores(self, score):
        """Normalize a score to the range [0, 1]."""
        normalized_score = (score - self.min_score) / (self.max_score - self.min_score)
        return normalized_score

    def complexity(self, sentence):
        """Measure and normalize the complexity of the input sentence."""
        raw_complexity = self.semantic_complexity(sentence)
        return self.normalize_scores(raw_complexity)

    def update_score_range(self, new_scores):
        """Update the score range based on new computed scores."""
        self.min_score = min(self.min_score, min(new_scores))
        self.max_score = max(self.max_score, max(new_scores))

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