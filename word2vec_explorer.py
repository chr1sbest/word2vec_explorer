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

class CommandHandler:
    """Handles parsing and execution of user commands"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def validate_words(self, *words):
        """Validate that all words exist in vocabulary"""
        if not self.model_manager.is_loaded():
            return False, "Model not loaded"

        missing = [w for w in words if not self.model_manager.word_exists(w)]
        if missing:
            return False, f"Words not in vocabulary: {', '.join(missing)}"
        return True, None

    def similar(self, word, n=10):
        """Find N most similar words to the given word"""
        valid, error = self.validate_words(word)
        if not valid:
            return {"success": False, "error": error}

        try:
            results = self.model_manager.model.most_similar(word, topn=n)
            return {
                "success": True,
                "word": word,
                "results": [{"word": w, "similarity": float(s)} for w, s in results]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analogy(self, word1, word2, word3, n=10):
        """Find words where word1:word2 :: word3:X"""
        valid, error = self.validate_words(word1, word2, word3)
        if not valid:
            return {"success": False, "error": error}

        try:
            # word2vec analogy: king - man + woman = queen
            results = self.model_manager.model.most_similar(
                positive=[word1, word3],
                negative=[word2],
                topn=n
            )
            return {
                "success": True,
                "analogy": f"{word1}:{word2} :: {word3}:?",
                "results": [{"word": w, "similarity": float(s)} for w, s in results]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def main():
    print("Word2Vec Explorer starting...")

if __name__ == "__main__":
    main()
