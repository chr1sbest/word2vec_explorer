#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings
"""

import gensim.downloader as api
from gensim.models import KeyedVectors

class ModelManager:
    """Manages word2vec model loading and operations"""

    def __init__(self, auto_load=True):
        self.model = None
        self._vocab = None
        if auto_load:
            self.load_model()

    def load_model(self):
        """Load pre-trained word2vec model from gensim"""
        print("Loading word2vec model (this takes 30-60 seconds on first run)...")
        try:
            self.model = api.load("word2vec-google-news-300")
            self._vocab = set(self.model.index_to_key)
            print(f"✓ Model loaded successfully! Vocabulary size: {len(self._vocab):,} words")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Run: python -c 'import gensim.downloader as api; api.load(\"word2vec-google-news-300\")'")
            raise

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def word_exists(self, word):
        """Check if word exists in vocabulary"""
        if not self.is_loaded():
            return False
        return word in self.model

def main():
    print("Word2Vec Explorer starting...")

if __name__ == "__main__":
    main()
