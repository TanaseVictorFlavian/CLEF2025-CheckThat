from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from typing import List


class Encoder(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def encode(self, data) -> np.ndarray:
        pass

    @abstractmethod
    def get_emb_dim(self) -> int:
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        pass

class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, data) -> np.ndarray:
        return self.model.encode(data, 
                                batch_size = 64, 
                                show_progress_bar=True)

    def get_emb_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def get_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.get_emb_dim()
        }

class Word2VecEncoder(Encoder):
    def __init__(self):
        super().__init__()
        from gensim.models import KeyedVectors
        self.model_name = "Word2Vec - GoogleNews-vectors-negative300"    
        self.model = KeyedVectors.load_word2vec_format('task1/pretrained_model_weights/GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.emb_dim = self.model.vector_size

    def encode(self, texts: List[str]) -> np.ndarray:
        def preprocess(text):
            return text.lower().split()

        embeddings = []
        for sentence in texts:
            tokens = preprocess(sentence)
            vectors = [self.model[token] for token in tokens if token in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.emb_dim))
        return np.stack(embeddings)

    def get_emb_dim(self) -> int:
        return self.emb_dim

    def get_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.emb_dim,
        }
