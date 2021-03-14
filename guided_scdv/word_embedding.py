from gensim.models import KeyedVectors, Word2Vec

from .document import Documents


class WordEmbedding:
    def __init__(self, documents: Documents) -> None:
        self.documents: Documents = documents
        self.model: Word2Vec = None
        self.embedding: KeyedVectors = None

    def train(self, params: dict) -> None:
        with Word2Vec(**params) as model:
            model.build_vocab(self.documents)
            model.train(
                self.documents,
                total_examples=model.corpus_count
            )
            self.embedding = model.wv

    def load(self, embedding: KeyedVectors) -> None:
        self.embedding = embedding
