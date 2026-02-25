#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings

Based on the paper "Efficient Estimation of Word Representations in Vector Space"
by Mikolov et al. (2013)
"""

import re
import urllib.request
import gensim.downloader as api
from gensim.models import KeyedVectors
import sys
import os
import threading
import time
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

# Metadata for supported models — avoids relying on gensim's api.info()
# which fetches a remote JSON file that can be unavailable.
MODEL_REGISTRY = {
    'fasttext-wiki-news-subwords-300': {
        'size_mb': 958,
        'num_records': 999_999,
        'description': '1M word vectors trained on Wikipedia 2017 + news (16B tokens)',
    },
    'glove-wiki-gigaword-100': {
        'size_mb': 128,
        'num_records': 400_000,
        'description': 'GloVe vectors trained on Wikipedia 2014 + Gigaword 5 (6B tokens)',
    },
    'word2vec-google-news-300': {
        'size_mb': 1_663,
        'num_records': 3_000_000,
        'description': 'Word2Vec vectors trained on Google News (~100B words)',
    },
}

# Base URL for model files — same source gensim uses internally
GENSIM_RELEASES = "https://github.com/RaRe-Technologies/gensim-data/releases/download"

class ModelManager:
    """Manages word2vec model loading and operations"""

    DEFAULT_MODEL = "fasttext-wiki-news-subwords-300"

    def __init__(self, auto_load=True, model_name=None):
        self.model = None
        self._vocab = None
        self.model_name = model_name or self.DEFAULT_MODEL
        if auto_load:
            self.load_model()

    def _download_model(self, model_dir):
        """Download model files directly from GitHub releases."""
        import shutil
        os.makedirs(model_dir, exist_ok=True)
        base_url = f"{GENSIM_RELEASES}/{self.model_name}"

        # Download __init__.py (needed by load_data())
        urllib.request.urlretrieve(
            f"{base_url}/__init__.py",
            os.path.join(model_dir, "__init__.py")
        )

        # Download model file with progress bar
        model_file = f"{self.model_name}.gz"
        model_path = os.path.join(model_dir, model_file)

        def show_progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size, 1.0)
                done_mb = count * block_size / 1024 ** 2
                total_mb = total_size / 1024 ** 2
                filled = int(pct * 40)
                bar = "█" * filled + "─" * (40 - filled)
                print(f"\r   [{bar}] {pct*100:.1f}% {done_mb:.0f}/{total_mb:.0f}MB", end='', flush=True)

        try:
            urllib.request.urlretrieve(f"{base_url}/{model_file}", model_path, reporthook=show_progress)
            print()  # newline after progress bar
        except Exception:
            # Clean up partial download
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir)
            raise

    def load_model(self):
        """Load pre-trained model"""
        model_dir = os.path.join(api.BASE_DIR, self.model_name)
        is_cached = os.path.isdir(model_dir)

        if is_cached:
            print(f"\n📥 Loading {self.model_name} from cache...")
            print(f"   (Initializing into memory: 2-5 minutes)\n")
        else:
            meta = MODEL_REGISTRY.get(self.model_name, {})
            size_mb = meta.get('size_mb', '?')
            print(f"\n📥 Downloading {self.model_name}...")
            print(f"   Size: ~{size_mb}MB")
            print(f"   Destination: {api.BASE_DIR}\n")
            try:
                self._download_model(model_dir)
                print(f"\n   ✓ Download complete. Initializing into memory (2-5 min)...\n")
            except Exception as e:
                print(f"\n✗ Download failed: {e}")
                print(f"   Please check your internet connection and try again.\n")
                raise

        # Spinner only during the memory-initialization phase
        stop_spinner = threading.Event()
        def show_spinner():
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            while not stop_spinner.is_set():
                print(f'\r⚙️  Initializing model {spinner_chars[idx % len(spinner_chars)]}', end='', flush=True)
                idx += 1
                time.sleep(0.1)
            print('\r' + ' ' * 50 + '\r', end='', flush=True)

        try:
            spinner_thread = threading.Thread(target=show_spinner, daemon=True)
            spinner_thread.start()

            sys.path.insert(0, api.BASE_DIR)
            module = __import__(self.model_name)
            self.model = module.load_data()

            stop_spinner.set()
            spinner_thread.join(timeout=0.5)

            self._vocab = set(self.model.index_to_key)
            print(f"✓ Ready! Vocabulary: {len(self._vocab):,} words\n")
        except Exception as e:
            stop_spinner.set()
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
        try:
            self.model_manager = ModelManager(model_name=self.model_name)
        except Exception:
            print("\nPlease check your internet connection and try again.")
            print("On first run, a network connection is required to download model data.\n")
            return
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
    """List supported pre-trained models"""
    print("\n📦 Supported Pre-trained Models\n" + "="*60 + "\n")

    categories = {'FastText': [], 'GloVe': [], 'Word2Vec': []}
    for name, meta in MODEL_REGISTRY.items():
        entry = (name, meta['size_mb'], meta['num_records'], meta['description'])
        if 'fasttext' in name:
            categories['FastText'].append(entry)
        elif 'glove' in name:
            categories['GloVe'].append(entry)
        elif 'word2vec' in name:
            categories['Word2Vec'].append(entry)

    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            for name, size_mb, vocab, desc in items:
                print(f"  • {name:45} {size_mb:5d}MB  {vocab:>12,} words")
                print(f"    {desc}")

    print(f"\n{'='*60}")
    print(f"Default: {ModelManager.DEFAULT_MODEL}")
    print("\nUsage: ./explore.sh --model <model-name>")
    print("Example: ./explore.sh --model glove-wiki-gigaword-100\n")


def select_model_interactive():
    """Interactive model selector with 3 recommended options"""
    print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}  Select Word Embedding Model{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")

    models = [
        {
            'name': 'fasttext-wiki-news-subwords-300',
            'display': 'FastText Wiki-News (Recommended)',
            'size': '958MB',
            'load_time': '2-5 min',
            'description': 'Best for cultural/food analogies (japan:sushi → canada:poutine)',
            'pros': '✓ Handles rare words  ✓ Best analogies  ✓ Moderate size'
        },
        {
            'name': 'glove-wiki-gigaword-100',
            'display': 'GloVe Wikipedia (Lightweight)',
            'size': '128MB',
            'load_time': '~30s',
            'description': 'Fast loading, good for basic exploration',
            'pros': '✓ Quick to load  ✓ Small download  ✓ Good accuracy'
        },
        {
            'name': 'word2vec-google-news-300',
            'display': 'Word2Vec Google News (Original)',
            'size': '1.6GB',
            'load_time': '3-8 min',
            'description': 'Classic model from the original 2013 paper',
            'pros': '✓ Largest vocabulary  ✓ News-focused  ✓ Most tested'
        }
    ]

    for i, model in enumerate(models, 1):
        print(f"{OutputFormatter.BOLD}{i}. {model['display']}{OutputFormatter.RESET}")
        print(f"   {model['description']}")
        print(f"   Size: {model['size']} | Load time: {model['load_time']}")
        print(f"   {OutputFormatter.GREEN}{model['pros']}{OutputFormatter.RESET}")
        print()

    while True:
        try:
            choice = input("Choose model (1-3) [default: 1]: ").strip()
            if not choice:
                choice = '1'

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected = models[choice_idx]
                print(f"\n✓ Selected: {selected['display']}")
                return selected['name']
            else:
                print(f"{OutputFormatter.RED}Please enter 1, 2, or 3{OutputFormatter.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{OutputFormatter.RED}Invalid choice. Please enter 1, 2, or 3{OutputFormatter.RESET}")
        except EOFError:
            print(f"\n{OutputFormatter.YELLOW}Using default (FastText){OutputFormatter.RESET}")
            return models[0]['name']

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
        help='Pre-trained model to use (skips interactive selection)'
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

    # If no model specified, show interactive selector
    model_name = args.model
    if model_name is None:
        model_name = select_model_interactive()

    repl = WordVecREPL(model_name=model_name)
    repl.start()

if __name__ == "__main__":
    main()
