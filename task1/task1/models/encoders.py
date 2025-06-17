from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch


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
        
class UmbertoEncoder(Encoder):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.emb_dim = self.model.config.hidden_size

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # explicitly truncate long sequences
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

    def get_emb_dim(self) -> int:
        return self.emb_dim

    def get_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.get_emb_dim()
        }
        
class ArabicBertEncoder(Encoder):
    def __init__(self, model_name: str = "asafaya/bert-base-arabic", device: str = "cpu", pooling: str = "cls"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.pooling = pooling  # "cls" or "mean"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.emb_dim = self.model.config.hidden_size

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state

            if self.pooling == "mean":
                # Mean pooling
                input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                # CLS token
                embeddings = last_hidden_state[:, 0, :]

        return embeddings.cpu().numpy()

    def get_emb_dim(self) -> int:
        return self.emb_dim

    def get_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.get_emb_dim(),
            "pooling": self.pooling
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

class TfidfEncoder(Encoder):
    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        **tfidf_kwargs
    ):
        super().__init__()
        self.model_name = "TFIDF encoder"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **tfidf_kwargs
        )
        self._fitted = False
        self.emb_dim: int | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        # On first call, fit the vectorizer; afterwards just transform
        if not self._fitted:
            X = self.vectorizer.fit_transform(texts)
            self.emb_dim = X.shape[1]
            self._fitted = True
        else:
            X = self.vectorizer.transform(texts)
        return X.toarray()

    def get_emb_dim(self) -> int:
        return self.emb_dim

    def get_params(self) -> dict:
        return {
            "model_name":    self.model_name,
            "max_features":  self.vectorizer.max_features,
            "ngram_range":   self.vectorizer.ngram_range,
            "embedding_dim": self.get_emb_dim(),
        }
