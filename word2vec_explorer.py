#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings

Based on the paper "Efficient Estimation of Word Representations in Vector Space"
by Mikolov et al. (2013)
"""

import re
import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm import tqdm
import sys
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

class ModelManager:
    """Manages word2vec model loading and operations"""

    DEFAULT_MODEL = "fasttext-wiki-news-subwords-300"

    def __init__(self, auto_load=True, model_name=None):
        self.model = None
        self._vocab = None
        self.model_name = model_name or self.DEFAULT_MODEL
        if auto_load:
            self.load_model()

    def load_model(self):
        """Load pre-trained model from gensim"""
        # Check if model is already cached
        is_cached = False
        try:
            info = api.info()
            model_info = info['models'].get(self.model_name, {})
            size_mb = model_info.get('file_size', 0) / (1024*1024)

            # Check if already downloaded
            model_dir = os.path.join(api.BASE_DIR, self.model_name)
            is_cached = os.path.exists(model_dir)

            if not is_cached:
                print(f"\n📥 Downloading {self.model_name}...")
                print(f"   Size: {size_mb:.0f}MB")
                print(f"   Destination: {api.BASE_DIR}")
                print(f"   (After download: ~30s to initialize into memory)\n")
            else:
                print(f"\n📥 Loading {self.model_name} from cache...")
                print(f"   (Initializing into memory: ~10-20s)\n")
        except Exception as e:
            # Fallback if info check fails, but still show time estimate
            print(f"\n📥 Loading {self.model_name}...")
            print(f"   (Download + initialization may take 2-3 minutes on first run)\n")

        try:
            # Suppress gensim's verbose output but keep progress bar
            class TqdmUpTo(tqdm):
                """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            # Load with cleaner progress
            original_stdout = sys.stdout

            # Load model (download if needed + initialize into memory)
            self.model = api.load(self.model_name)

            self._vocab = set(self.model.index_to_key)

            print(f"✓ Ready! Vocabulary: {len(self._vocab):,} words\n")
        except Exception as e:
            print(f"\n✗ Error loading model: {e}")
            print(f"   Run with --list-models to see available options\n")
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
        """Find words where word1:word2 :: word3:X

        Uses word2vec formula: word3 - word1 + word2 = X
        Example: man:woman :: uncle:? => uncle - man + woman = aunt
        """
        valid, error = self.validate_words(word1, word2, word3)
        if not valid:
            return {"success": False, "error": error}

        try:
            # Formula: word3 - word1 + word2 = X
            # Example: uncle - man + woman = aunt
            results = self.model_manager.model.most_similar(
                positive=[word3, word2],  # word3 + word2
                negative=[word1],          # - word1
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

class WordVecREPL:
    """Interactive REPL for word2vec exploration"""

    COMMANDS = ['analogy', 'similar', 'distance', 'find', 'vector', 'help', 'quit', 'exit']

    def __init__(self, model_name=None):
        self.model_manager = None
        self.command_handler = None
        self.formatter = OutputFormatter()
        self.model_name = model_name
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.COMMANDS, ignore_case=True)
        )

    def start(self):
        """Start the REPL"""
        self.print_welcome()

        # Load model
        self.model_manager = ModelManager(model_name=self.model_name)
        self.command_handler = CommandHandler(self.model_manager)

        # Main loop
        while True:
            try:
                user_input = self.session.prompt('\nword2vec> ').strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'help':
                    self.print_help()
                    continue

                self.execute_command(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                print("\nGoodbye!")
                break

    def print_welcome(self):
        """Print welcome message"""
        print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}  Word2Vec Explorer - Interactive Embedding Explorer{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")
        print("Type 'help' for available commands or 'quit' to exit.\n")

    def print_help(self):
        """Print help message"""
        help_text = f"""
{OutputFormatter.BOLD}Available Commands:{OutputFormatter.RESET}

  {OutputFormatter.BLUE}analogy{OutputFormatter.RESET} word1:word2 word3:
      Find X where word1:word2 :: word3:X
      Example: analogy king:man woman:
               (finds "queen" - king is to man as woman is to ?)

  {OutputFormatter.BLUE}similar{OutputFormatter.RESET} word [n]
      Find N most similar words (default n=10)
      Example: similar python 5

  {OutputFormatter.BLUE}distance{OutputFormatter.RESET} word1 word2
      Calculate cosine similarity between two words
      Example: distance cat dog

  {OutputFormatter.BLUE}find{OutputFormatter.RESET} pattern
      Search vocabulary (use * as wildcard)
      Example: find prog*

  {OutputFormatter.BLUE}vector{OutputFormatter.RESET} word
      Display the embedding vector for a word
      Example: vector king

  {OutputFormatter.BLUE}help{OutputFormatter.RESET}
      Show this help message

  {OutputFormatter.BLUE}quit{OutputFormatter.RESET} / {OutputFormatter.BLUE}exit{OutputFormatter.RESET}
      Exit the explorer
"""
        print(help_text)

    def execute_command(self, user_input):
        """Parse and execute user command"""
        parts = user_input.split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == 'analogy':
            # Support both formats: "analogy king:man woman:" and "analogy king man woman"
            if len(args) < 1:
                print(f"{OutputFormatter.RED}Usage: analogy word1:word2 word3: [n]{OutputFormatter.RESET}")
                print(f"{OutputFormatter.YELLOW}Example: analogy king:man woman:{OutputFormatter.RESET}")
                return

            # Parse colon format: "king:man woman:"
            if ':' in args[0]:
                try:
                    pair1 = args[0].rstrip(':').split(':')
                    if len(pair1) != 2:
                        print(f"{OutputFormatter.RED}Invalid format. Use: word1:word2 word3:{OutputFormatter.RESET}")
                        return
                    word1, word2 = pair1[0], pair1[1]

                    if len(args) < 2:
                        print(f"{OutputFormatter.RED}Usage: analogy word1:word2 word3: [n]{OutputFormatter.RESET}")
                        return

                    word3 = args[1].rstrip(':')
                    n = int(args[2]) if len(args) > 2 else 10
                except (ValueError, IndexError):
                    print(f"{OutputFormatter.RED}Invalid format. Example: analogy king:man woman:{OutputFormatter.RESET}")
                    return
            # Support old format for backwards compatibility
            elif len(args) >= 3:
                word1, word2, word3 = args[0], args[1], args[2]
                n = int(args[3]) if len(args) > 3 else 10
            else:
                print(f"{OutputFormatter.RED}Usage: analogy word1:word2 word3: [n]{OutputFormatter.RESET}")
                print(f"{OutputFormatter.YELLOW}Example: analogy king:man woman:{OutputFormatter.RESET}")
                return

            result = self.command_handler.analogy(word1, word2, word3, n)
            print(self.formatter.format_analogy(result))

        elif cmd == 'similar':
            if len(args) < 1:
                print(f"{OutputFormatter.RED}Usage: similar word [n]{OutputFormatter.RESET}")
                return
            n = int(args[1]) if len(args) > 1 else 10
            result = self.command_handler.similar(args[0], n)
            print(self.formatter.format_similar(result))

        elif cmd == 'distance':
            if len(args) < 2:
                print(f"{OutputFormatter.RED}Usage: distance word1 word2{OutputFormatter.RESET}")
                return
            result = self.command_handler.distance(args[0], args[1])
            print(self.formatter.format_distance(result))

        elif cmd == 'find':
            if len(args) < 1:
                print(f"{OutputFormatter.RED}Usage: find pattern{OutputFormatter.RESET}")
                return
            result = self.command_handler.find(args[0])
            print(self.formatter.format_find(result))

        elif cmd == 'vector':
            if len(args) < 1:
                print(f"{OutputFormatter.RED}Usage: vector word{OutputFormatter.RESET}")
                return
            result = self.command_handler.vector(args[0])
            print(self.formatter.format_vector(result))

        else:
            print(f"{OutputFormatter.RED}Unknown command: {cmd}{OutputFormatter.RESET}")
            print("Type 'help' for available commands.")

def list_available_models():
    """List all available pre-trained models"""
    print("\n📦 Available Pre-trained Models\n" + "="*60 + "\n")

    models = api.info()['models']

    # Categorize models
    categories = {
        'Word2Vec': [],
        'GloVe': [],
        'FastText': [],
        'Other': []
    }

    for name, info in models.items():
        size_mb = info.get('file_size', 0) / (1024*1024)
        vocab = info.get('num_records', 'unknown')

        if 'word2vec' in name:
            categories['Word2Vec'].append((name, size_mb, vocab))
        elif 'glove' in name:
            categories['GloVe'].append((name, size_mb, vocab))
        elif 'fasttext' in name:
            categories['FastText'].append((name, size_mb, vocab))
        elif 'testing' not in name:
            categories['Other'].append((name, size_mb, vocab))

    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            for name, size_mb, vocab in items:
                vocab_str = f"{vocab:,}" if isinstance(vocab, int) else str(vocab)
                print(f"  • {name:45} {size_mb:6.0f}MB  {vocab_str:>12} words")

    print(f"\n{'='*60}")
    print("Default: word2vec-google-news-300")
    print("\nUsage: ./explore.sh --model <model-name>")
    print("Example: ./explore.sh --model glove-twitter-200\n")


def main():
    """Entry point for word2vec explorer"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Word2Vec Explorer - Interactive REPL for exploring word embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Pre-trained model to use (default: word2vec-google-news-300)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available pre-trained models and exit'
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    repl = WordVecREPL(model_name=args.model)
    repl.start()

if __name__ == "__main__":
    main()
