#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings
"""

import re
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

    def distance(self, word1, word2):
        """Calculate cosine similarity between two words"""
        valid, error = self.validate_words(word1, word2)
        if not valid:
            return {"success": False, "error": error}

        try:
            similarity = self.model_manager.model.similarity(word1, word2)
            return {
                "success": True,
                "word1": word1,
                "word2": word2,
                "similarity": float(similarity)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def find(self, pattern, max_results=20):
        """Search vocabulary for words matching pattern (supports * wildcard)"""
        if not self.model_manager.is_loaded():
            return {"success": False, "error": "Model not loaded"}

        try:
            # Convert wildcard pattern to regex
            regex_pattern = pattern.replace('*', '.*')
            regex = re.compile(f"^{regex_pattern}$", re.IGNORECASE)

            matches = [w for w in self.model_manager._vocab if regex.match(w)]
            matches = sorted(matches)[:max_results]

            return {
                "success": True,
                "pattern": pattern,
                "matches": matches,
                "total": len(matches),
                "truncated": len(matches) == max_results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def vector(self, word):
        """Get the embedding vector for a word"""
        valid, error = self.validate_words(word)
        if not valid:
            return {"success": False, "error": error}

        try:
            vec = self.model_manager.model[word]
            return {
                "success": True,
                "word": word,
                "vector": vec.tolist(),
                "dimensions": len(vec)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class OutputFormatter:
    """Formats command results for terminal display"""

    # ANSI color codes
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    @staticmethod
    def format_similar(result):
        """Format similar command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        lines = [f"\n{OutputFormatter.BOLD}Most similar to '{result['word']}':{OutputFormatter.RESET}\n"]
        for i, item in enumerate(result["results"], 1):
            score = item["similarity"]
            word = item["word"]
            bar = "█" * int(score * 20)
            lines.append(f"  {i:2}. {word:20} {OutputFormatter.GREEN}{score:.4f}{OutputFormatter.RESET} {bar}")
        return "\n".join(lines)

    @staticmethod
    def format_analogy(result):
        """Format analogy command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        lines = [f"\n{OutputFormatter.BOLD}{result['analogy']}{OutputFormatter.RESET}\n"]
        for i, item in enumerate(result["results"], 1):
            score = item["similarity"]
            word = item["word"]
            bar = "█" * int(score * 20)
            lines.append(f"  {i:2}. {word:20} {OutputFormatter.GREEN}{score:.4f}{OutputFormatter.RESET} {bar}")
        return "\n".join(lines)

    @staticmethod
    def format_distance(result):
        """Format distance command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        score = result["similarity"]
        bar = "█" * int(score * 40)
        return (f"\n{OutputFormatter.BOLD}Distance between '{result['word1']}' and '{result['word2']}':{OutputFormatter.RESET}\n"
                f"  Similarity: {OutputFormatter.GREEN}{score:.4f}{OutputFormatter.RESET} {bar}\n")

    @staticmethod
    def format_find(result):
        """Format find command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        if not result["matches"]:
            return f"\n{OutputFormatter.YELLOW}No matches found for pattern '{result['pattern']}'{OutputFormatter.RESET}\n"

        lines = [f"\n{OutputFormatter.BOLD}Matches for '{result['pattern']}':{OutputFormatter.RESET}\n"]
        for word in result["matches"]:
            lines.append(f"  • {word}")

        if result["truncated"]:
            lines.append(f"\n  {OutputFormatter.YELLOW}(showing first {len(result['matches'])} results){OutputFormatter.RESET}")

        return "\n".join(lines)

    @staticmethod
    def format_vector(result):
        """Format vector command results"""
        if not result["success"]:
            return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

        vec = result["vector"]
        lines = [f"\n{OutputFormatter.BOLD}Vector for '{result['word']}' ({result['dimensions']} dimensions):{OutputFormatter.RESET}\n"]

        # Show first 10 and last 10 values
        lines.append(f"  First 10: {OutputFormatter.BLUE}{vec[:10]}{OutputFormatter.RESET}")
        lines.append(f"  Last 10:  {OutputFormatter.BLUE}{vec[-10:]}{OutputFormatter.RESET}")

        # Show statistics
        import numpy as np
        vec_array = np.array(vec)
        lines.append(f"\n  Stats: min={vec_array.min():.4f}, max={vec_array.max():.4f}, "
                    f"mean={vec_array.mean():.4f}, std={vec_array.std():.4f}\n")

        return "\n".join(lines)

def main():
    print("Word2Vec Explorer starting...")

if __name__ == "__main__":
    main()
