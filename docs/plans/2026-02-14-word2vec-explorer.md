# Word2Vec Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive REPL for exploring word2vec embeddings with analogy, similarity, distance, search, and vector inspection operations.

**Architecture:** Three-layer design: ModelManager (gensim wrapper), CommandHandler (operations), REPL (prompt_toolkit interface). Use pre-trained Google News word2vec model with rich terminal output.

**Tech Stack:** Python 3.8+, gensim, prompt_toolkit, numpy

---

## Task 1: Set Up Project Structure

**Files:**
- Create: `requirements.txt`
- Create: `word2vec_explorer.py`
- Create: `tests/test_model_manager.py`

**Step 1: Create requirements.txt**

```txt
gensim>=4.3.0
prompt_toolkit>=3.0.0
numpy>=1.21.0
```

**Step 2: Create initial main file structure**

Create `word2vec_explorer.py`:

```python
#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings
"""

def main():
    print("Word2Vec Explorer starting...")

if __name__ == "__main__":
    main()
```

**Step 3: Make executable**

Run: `chmod +x word2vec_explorer.py`

**Step 4: Create tests directory**

Run: `mkdir -p tests && touch tests/__init__.py`

**Step 5: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 6: Commit**

```bash
git add requirements.txt word2vec_explorer.py tests/
git commit -m "feat: initialize project structure and dependencies"
```

---

## Task 2: Create ModelManager Component (TDD)

**Files:**
- Modify: `word2vec_explorer.py`
- Create: `tests/test_model_manager.py`

**Step 1: Write test for ModelManager initialization**

Add to `tests/test_model_manager.py`:

```python
import pytest
from word2vec_explorer import ModelManager

def test_model_manager_can_be_created():
    """Test that ModelManager can be instantiated without loading model"""
    manager = ModelManager(auto_load=False)
    assert manager is not None
    assert manager.model is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_manager.py::test_model_manager_can_be_created -v`
Expected: FAIL with "cannot import name 'ModelManager'"

**Step 3: Implement minimal ModelManager**

Add to `word2vec_explorer.py` before `main()`:

```python
class ModelManager:
    """Manages word2vec model loading and operations"""

    def __init__(self, auto_load=True):
        self.model = None
        self._vocab = None
        if auto_load:
            self.load_model()

    def load_model(self):
        """Load pre-trained word2vec model from gensim"""
        pass  # Will implement in next step

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def word_exists(self, word):
        """Check if word exists in vocabulary"""
        if not self.is_loaded():
            return False
        return word in self.model
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_manager.py::test_model_manager_can_be_created -v`
Expected: PASS

**Step 5: Write test for word existence check**

Add to `tests/test_model_manager.py`:

```python
def test_word_exists_returns_false_when_not_loaded():
    """Test word_exists returns False when model not loaded"""
    manager = ModelManager(auto_load=False)
    assert manager.word_exists("test") is False
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_model_manager.py -v`
Expected: Both tests PASS

**Step 7: Commit**

```bash
git add word2vec_explorer.py tests/test_model_manager.py
git commit -m "feat: add ModelManager with vocabulary checking"
```

---

## Task 3: Implement Model Loading

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Add model loading with progress**

Update `load_model()` in `word2vec_explorer.py`:

```python
import gensim.downloader as api
from gensim.models import KeyedVectors

class ModelManager:
    # ... existing code ...

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
```

**Step 2: Test manually (this downloads the model)**

Run: `python3 -c "from word2vec_explorer import ModelManager; m = ModelManager()"`
Expected: Downloads model (~1.5GB), then prints success message with vocab size

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: implement word2vec model loading with progress"
```

---

## Task 4: Create CommandHandler Base

**Files:**
- Modify: `word2vec_explorer.py`
- Create: `tests/test_command_handler.py`

**Step 1: Write test for CommandHandler**

Create `tests/test_command_handler.py`:

```python
import pytest
from word2vec_explorer import CommandHandler, ModelManager

def test_command_handler_creation():
    """Test CommandHandler can be created with ModelManager"""
    model_manager = ModelManager(auto_load=False)
    handler = CommandHandler(model_manager)
    assert handler is not None
    assert handler.model_manager == model_manager
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_command_handler_creation -v`
Expected: FAIL with "cannot import name 'CommandHandler'"

**Step 3: Implement CommandHandler base**

Add to `word2vec_explorer.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_command_handler_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add word2vec_explorer.py tests/test_command_handler.py
git commit -m "feat: add CommandHandler with word validation"
```

---

## Task 5: Implement Similar Command

**Files:**
- Modify: `word2vec_explorer.py`
- Modify: `tests/test_command_handler.py`

**Step 1: Write test for similar command**

Add to `tests/test_command_handler.py`:

```python
def test_similar_validates_word_exists():
    """Test similar command validates word"""
    model_manager = ModelManager(auto_load=False)
    handler = CommandHandler(model_manager)

    result = handler.similar("test", n=5)
    assert result["success"] is False
    assert "not loaded" in result["error"].lower() or "not in vocabulary" in result["error"].lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_similar_validates_word_exists -v`
Expected: FAIL with "CommandHandler has no attribute 'similar'"

**Step 3: Implement similar command**

Add to `CommandHandler` class:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_similar_validates_word_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add word2vec_explorer.py tests/test_command_handler.py
git commit -m "feat: implement similar command with validation"
```

---

## Task 6: Implement Analogy Command

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement analogy command**

Add to `CommandHandler` class:

```python
def analogy(self, word1, word2, word3, n=10):
    """Find words where word1:word2 :: word3:X"""
    valid, error = self.validate_words(word1, word2, word3)
    if not valid:
        return {"success": False, "error": error}

    try:
        # word2vec analogy: king - man + woman = queen
        results = self.model_manager.model.most_similar(
            positive=[word2, word3],
            negative=[word1],
            topn=n
        )
        return {
            "success": True,
            "analogy": f"{word1}:{word2} :: {word3}:?",
            "results": [{"word": w, "similarity": float(s)} for w, s in results]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Step 2: Test manually with loaded model**

Run:
```python
python3 -c "
from word2vec_explorer import ModelManager, CommandHandler
m = ModelManager()
h = CommandHandler(m)
result = h.analogy('king', 'man', 'woman', n=3)
print(result)
"
```
Expected: Returns dict with success=True and "queen" in top results

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: implement analogy command (king-man+woman=queen)"
```

---

## Task 7: Implement Distance Command

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement distance command**

Add to `CommandHandler` class:

```python
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
```

**Step 2: Test manually**

Run:
```python
python3 -c "
from word2vec_explorer import ModelManager, CommandHandler
m = ModelManager()
h = CommandHandler(m)
print(h.distance('cat', 'dog'))
print(h.distance('cat', 'France'))
"
```
Expected: cat/dog has higher similarity than cat/France

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: implement distance command for cosine similarity"
```

---

## Task 8: Implement Find Command

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement find command with pattern matching**

Add to `CommandHandler` class:

```python
import re

class CommandHandler:
    # ... existing code ...

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
```

**Step 2: Test manually**

Run:
```python
python3 -c "
from word2vec_explorer import ModelManager, CommandHandler
m = ModelManager()
h = CommandHandler(m)
print(h.find('prog*', max_results=10))
"
```
Expected: Returns words like "program", "programming", "programmer", etc.

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: implement find command with wildcard search"
```

---

## Task 9: Implement Vector Command

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement vector command**

Add to `CommandHandler` class:

```python
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
```

**Step 2: Test manually**

Run:
```python
python3 -c "
from word2vec_explorer import ModelManager, CommandHandler
m = ModelManager()
h = CommandHandler(m)
result = h.vector('king')
print(f\"Dimensions: {result['dimensions']}\")
print(f\"First 5 values: {result['vector'][:5]}\")
"
```
Expected: Shows 300 dimensions and first 5 float values

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: implement vector command to inspect embeddings"
```

---

## Task 10: Create OutputFormatter

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement OutputFormatter class**

Add to `word2vec_explorer.py`:

```python
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
```

**Step 2: Test manually**

Run:
```python
python3 -c "
from word2vec_explorer import ModelManager, CommandHandler, OutputFormatter
m = ModelManager()
h = CommandHandler(m)
result = h.similar('king', n=5)
print(OutputFormatter.format_similar(result))
"
```
Expected: Shows colored, formatted output with bars

**Step 3: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: add OutputFormatter with colored terminal output"
```

---

## Task 11: Create REPL Interface

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Implement REPL class with prompt_toolkit**

Add to `word2vec_explorer.py`:

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

class WordVecREPL:
    """Interactive REPL for word2vec exploration"""

    COMMANDS = ['analogy', 'similar', 'distance', 'find', 'vector', 'help', 'quit', 'exit']

    def __init__(self):
        self.model_manager = None
        self.command_handler = None
        self.formatter = OutputFormatter()
        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.COMMANDS, ignore_case=True)
        )

    def start(self):
        """Start the REPL"""
        self.print_welcome()

        # Load model
        self.model_manager = ModelManager()
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

  {OutputFormatter.BLUE}analogy{OutputFormatter.RESET} word1 word2 word3
      Find X where word1:word2 :: word3:X
      Example: analogy king man woman
               (finds "queen" - king is to man as woman is to queen)

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
            if len(args) < 3:
                print(f"{OutputFormatter.RED}Usage: analogy word1 word2 word3 [n]{OutputFormatter.RESET}")
                return
            n = int(args[3]) if len(args) > 3 else 10
            result = self.command_handler.analogy(args[0], args[1], args[2], n)
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
```

**Step 2: Update main() function**

Replace the `main()` function:

```python
def main():
    """Entry point for word2vec explorer"""
    repl = WordVecREPL()
    repl.start()
```

**Step 3: Test the REPL**

Run: `python3 word2vec_explorer.py`

Try commands:
1. `help` - shows help
2. `similar king 5` - shows words similar to king
3. `analogy king man woman` - shows queen
4. `distance cat dog` - shows similarity
5. `find prog*` - finds words starting with prog
6. `vector king` - shows embedding
7. `quit` - exits

Expected: All commands work with colored output

**Step 4: Commit**

```bash
git add word2vec_explorer.py
git commit -m "feat: add interactive REPL with prompt_toolkit"
```

---

## Task 12: Create README Documentation

**Files:**
- Create: `README.md`

**Step 1: Write comprehensive README**

Create `README.md`:

```markdown
# Word2Vec Explorer

An interactive Python REPL for exploring word embeddings using the classic word2vec model. Perfect for learning about vector semantics and demonstrating the famous "king - man + woman = queen" analogy.

## Features

- **Analogy Exploration**: Find analogical relationships (e.g., king:man :: woman:?)
- **Similarity Search**: Find words most similar to a given word
- **Distance Calculation**: Compute cosine similarity between word pairs
- **Vocabulary Search**: Find words matching patterns (with wildcard support)
- **Vector Inspection**: View the actual embedding vectors
- **Rich REPL**: Command history, auto-suggestions, and colored output

## Installation

### Requirements

- Python 3.8 or higher
- ~2GB disk space for the pre-trained model

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the explorer (downloads model on first run - takes 30-60 seconds)
python3 word2vec_explorer.py
```

## Usage

### Available Commands

#### Analogy: Find Analogical Relationships

```
word2vec> analogy king man woman
```

Finds words where "king is to man as woman is to X" (typically finds "queen")

#### Similar: Find Similar Words

```
word2vec> similar python 10
```

Shows the 10 most similar words to "python"

#### Distance: Calculate Similarity

```
word2vec> distance cat dog
```

Shows the cosine similarity score between "cat" and "dog"

#### Find: Search Vocabulary

```
word2vec> find prog*
```

Finds all words starting with "prog" (program, programming, etc.)

#### Vector: Inspect Embeddings

```
word2vec> vector king
```

Displays the 300-dimensional embedding vector for "king"

### Example Session

```
word2vec> analogy king man woman

king:man :: woman:?

   1. queen                0.7118 ████████████████
   2. monarch              0.6189 ████████████
   3. princess             0.5902 ███████████
   4. crown_prince         0.5499 ██████████

word2vec> similar queen 5

Most similar to 'queen':

   1. princess             0.6510 █████████████
   2. monarch              0.6413 ████████████
   3. throne               0.5964 ███████████
   4. elizabeth            0.5896 ███████████
   5. royal                0.5643 ███████████

word2vec> distance cat dog

Distance between 'cat' and 'dog':
  Similarity: 0.7608 ██████████████████████████████
```

## How It Works

This tool uses Google's pre-trained word2vec model (300-dimensional vectors trained on ~100 billion words from Google News). The model represents each word as a 300-dimensional vector where semantically similar words have similar vectors.

The famous "king - man + woman = queen" example works because:
- The vector for "king" contains both "royalty" and "male" semantic components
- Subtracting "man" removes the "male" component
- Adding "woman" adds the "female" component
- The result is closest to "queen" (royalty + female)

## Technical Details

- **Model**: word2vec-google-news-300 (from gensim)
- **Architecture**: Skip-gram with negative sampling
- **Vocabulary**: ~3 million words
- **Vector dimensions**: 300
- **Similarity metric**: Cosine similarity

## Tips

- Words are case-sensitive (try "Paris" not "paris")
- Multi-word phrases use underscores (e.g., "New_York")
- Use wildcards in `find` command to explore the vocabulary
- The `vector` command shows statistics (min, max, mean, std)

## References

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (original word2vec paper)
- [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)
- [Gensim documentation](https://radimrehurek.com/gensim/)

## License

Educational project - see original word2vec and gensim licenses for model usage.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with examples"
```

---

## Task 13: Final Testing and Polish

**Files:**
- Modify: `word2vec_explorer.py`

**Step 1: Add shebang and docstring**

Ensure top of `word2vec_explorer.py` has:

```python
#!/usr/bin/env python3
"""
Word2Vec Explorer - Interactive REPL for exploring word embeddings

Based on the paper "Efficient Estimation of Word Representations in Vector Space"
by Mikolov et al. (2013)
"""
```

**Step 2: Run full integration test**

Run: `python3 word2vec_explorer.py`

Test all commands:
1. `help`
2. `analogy king man woman`
3. `similar queen 5`
4. `distance cat dog`
5. `find king*`
6. `vector king`
7. Test error cases: `similar nonexistentword123`
8. `quit`

Expected: All commands work, errors handled gracefully, output is colored and formatted

**Step 3: Check code quality**

Run: `python3 -m py_compile word2vec_explorer.py`
Expected: No syntax errors

**Step 4: Final commit**

```bash
git add word2vec_explorer.py
git commit -m "polish: add final docstring and validation"
```

---

## Completion Checklist

- [ ] All 5 commands implemented (analogy, similar, distance, find, vector)
- [ ] Rich REPL with prompt_toolkit (history, suggestions, completion)
- [ ] Colored output with formatted results
- [ ] Error handling for missing words and invalid input
- [ ] README with examples and documentation
- [ ] Manual testing of all commands successful
- [ ] Clean git history with descriptive commits

## Success Criteria

1. Run `python3 word2vec_explorer.py` successfully
2. Execute `analogy king man woman` and see "queen" in results
3. All commands respond with colored, formatted output
4. Invalid words show helpful error messages
5. README provides clear usage instructions
