# GPT Token Explorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive REPL tool demonstrating how GPT models generate tokens through next-token prediction.

**Architecture:** Single-file design (gpt_token_explorer.py) with 4 classes: ModelManager (HF transformers), CommandHandler (inference), OutputFormatter (display), TokenREPL (interface).

**Tech Stack:** transformers, torch, prompt_toolkit, numpy, tqdm

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `explore.sh`
- Create: `.gitignore`

**Step 1: Create requirements.txt**

```txt
transformers>=4.30.0
torch>=2.0.0
prompt_toolkit>=3.0.0
numpy>=1.21.0
tqdm>=4.62.0
```

**Step 2: Create explore.sh launch script**

```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 gpt_token_explorer.py "$@"
```

Run: `chmod +x explore.sh`

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
venv/
.DS_Store
*.log
```

**Step 4: Commit**

```bash
git add requirements.txt explore.sh .gitignore
git commit -m "chore: initial project setup

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: ModelManager Class - Basic Structure

**Files:**
- Create: `gpt_token_explorer.py`
- Create: `tests/test_model_manager.py`

**Step 1: Write failing test for ModelManager creation**

Create `tests/test_model_manager.py`:
```python
import pytest
from gpt_token_explorer import ModelManager

def test_model_manager_can_be_created_without_loading():
    """ModelManager can be instantiated without auto-loading"""
    manager = ModelManager(auto_load=False)
    assert manager is not None
    assert manager.model is None
    assert manager.tokenizer is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_manager.py::test_model_manager_can_be_created_without_loading -v`
Expected: ImportError (module doesn't exist yet)

**Step 3: Write minimal ModelManager implementation**

Create `gpt_token_explorer.py`:
```python
#!/usr/bin/env python3
"""
GPT Token Explorer - Interactive REPL for learning token generation

Educational tool demonstrating next-token prediction, probability distributions,
and autoregressive generation in GPT models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import threading
import time
from tqdm import tqdm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
import numpy as np

class ModelManager:
    """Manages HuggingFace transformer model loading and operations"""

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M"

    def __init__(self, auto_load=True, model_name=None):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name or self.DEFAULT_MODEL
        if auto_load:
            self.load_model()

    def load_model(self):
        """Load model and tokenizer from HuggingFace"""
        pass  # Will implement in next task

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_manager.py::test_model_manager_can_be_created_without_loading -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_model_manager.py
git commit -m "feat: add ModelManager basic structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: ModelManager - Model Loading

**Files:**
- Modify: `gpt_token_explorer.py` (ModelManager class)
- Modify: `tests/test_model_manager.py`

**Step 1: Write test for model loading**

Add to `tests/test_model_manager.py`:
```python
def test_model_manager_loads_model():
    """ModelManager loads model and tokenizer"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    assert manager.is_loaded()
    assert manager.model is not None
    assert manager.tokenizer is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_manager.py::test_model_manager_loads_model -v`
Expected: FAIL (load_model() is not implemented)

**Step 3: Implement load_model() with progress indication**

Update `ModelManager.load_model()`:
```python
def load_model(self):
    """Load model and tokenizer from HuggingFace"""
    # Check if model is cached
    is_cached = self._check_cache()

    if not is_cached:
        print(f"\n📥 Downloading {self.model_name}...")
        print(f"   This may take a few minutes on first run")
    else:
        print(f"\n📥 Loading {self.model_name} from cache...")

    # Show spinner during loading
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(
        target=self._show_spinner,
        args=(stop_spinner, "Initializing model"),
        daemon=True
    )
    spinner_thread.start()

    try:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()  # Set to evaluation mode

        stop_spinner.set()
        spinner_thread.join(timeout=0.5)

        vocab_size = len(self.tokenizer)
        print(f"\n✓ Ready! Vocabulary: {vocab_size:,} tokens\n")

    except Exception as e:
        stop_spinner.set()
        spinner_thread.join(timeout=0.5)
        print(f"\n✗ Error loading model: {e}")
        raise

def _check_cache(self):
    """Check if model is in HuggingFace cache"""
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface"
    # Simple check - HF handles the details
    return cache_dir.exists()

def _show_spinner(self, stop_event, message):
    """Show spinner animation during model loading"""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    idx = 0
    while not stop_event.is_set():
        print(f'\r⚙️  {message} {spinner_chars[idx % len(spinner_chars)]}',
              end='', flush=True)
        idx += 1
        time.sleep(0.1)
    print('\r' + ' ' * 50 + '\r', end='', flush=True)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_manager.py::test_model_manager_loads_model -v -s`
Expected: PASS (may take time to download on first run)

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_model_manager.py
git commit -m "feat: implement model loading with progress indication

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: CommandHandler - Basic Structure

**Files:**
- Modify: `gpt_token_explorer.py`
- Create: `tests/test_command_handler.py`

**Step 1: Write test for CommandHandler creation**

Create `tests/test_command_handler.py`:
```python
import pytest
from gpt_token_explorer import ModelManager, CommandHandler

def test_command_handler_can_be_created():
    """CommandHandler can be instantiated with ModelManager"""
    manager = ModelManager(auto_load=False)
    handler = CommandHandler(manager)
    assert handler is not None
    assert handler.model_manager is manager
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_command_handler_can_be_created -v`
Expected: ImportError (CommandHandler doesn't exist)

**Step 3: Add CommandHandler class**

Add to `gpt_token_explorer.py` after ModelManager:
```python
class CommandHandler:
    """Handles command execution and inference operations"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def validate_input(self, text):
        """Validate user input"""
        if not text or not text.strip():
            return False, "Empty input. Please provide text."
        if not self.model_manager.is_loaded():
            return False, "Model not loaded."
        return True, None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_command_handler_can_be_created -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_command_handler.py
git commit -m "feat: add CommandHandler basic structure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: CommandHandler - Complete Command

**Files:**
- Modify: `gpt_token_explorer.py` (CommandHandler)
- Modify: `tests/test_command_handler.py`

**Step 1: Write test for complete command**

Add to `tests/test_command_handler.py`:
```python
def test_complete_returns_probabilities():
    """Complete command returns top-k token probabilities"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.complete("The capital of France is", top_k=5)

    assert result["success"] is True
    assert "results" in result
    assert len(result["results"]) == 5
    assert all("token" in r and "probability" in r for r in result["results"])
    # Probabilities should be in descending order
    probs = [r["probability"] for r in result["results"]]
    assert probs == sorted(probs, reverse=True)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_complete_returns_probabilities -v`
Expected: FAIL (complete() method doesn't exist)

**Step 3: Implement complete() method**

Add to `CommandHandler`:
```python
def complete(self, text, top_k=10):
    """Get next token probabilities

    Args:
        text: Input text prompt
        top_k: Number of top predictions to return

    Returns:
        dict with success, prompt, and results list
    """
    valid, error = self.validate_input(text)
    if not valid:
        return {"success": False, "error": error}

    try:
        # Tokenize input
        input_ids = self.model_manager.tokenizer.encode(
            text,
            return_tensors='pt'
        )

        # Get model output
        with torch.no_grad():
            outputs = self.model_manager.model(input_ids)

        # Extract logits for last position
        logits = outputs.logits[:, -1, :]

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get top-k
        top_k_probs, top_k_ids = torch.topk(probs[0], top_k)

        # Decode tokens
        results = []
        for prob, token_id in zip(top_k_probs, top_k_ids):
            token = self.model_manager.tokenizer.decode([token_id])
            results.append({
                "token": token,
                "token_id": int(token_id),
                "probability": float(prob)
            })

        return {
            "success": True,
            "prompt": text,
            "results": results
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_complete_returns_probabilities -v -s`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_command_handler.py
git commit -m "feat: implement complete command for next-token prediction

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: CommandHandler - Tokenize Command

**Files:**
- Modify: `gpt_token_explorer.py` (CommandHandler)
- Modify: `tests/test_command_handler.py`

**Step 1: Write test for tokenize command**

Add to `tests/test_command_handler.py`:
```python
def test_tokenize_breaks_text_into_tokens():
    """Tokenize command breaks text into tokens with IDs"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.tokenize("Hello world!")

    assert result["success"] is True
    assert "tokens" in result
    assert len(result["tokens"]) > 0
    assert all("text" in t and "token_id" in t for t in result["tokens"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_tokenize_breaks_text_into_tokens -v`
Expected: FAIL

**Step 3: Implement tokenize() method**

Add to `CommandHandler`:
```python
def tokenize(self, text):
    """Break text into tokens with IDs

    Args:
        text: Input text to tokenize

    Returns:
        dict with success, text, and tokens list
    """
    valid, error = self.validate_input(text)
    if not valid:
        return {"success": False, "error": error}

    try:
        # Encode to get token IDs
        token_ids = self.model_manager.tokenizer.encode(text)

        # Decode each token individually
        tokens = []
        for token_id in token_ids:
            token_text = self.model_manager.tokenizer.decode([token_id])
            tokens.append({
                "text": token_text,
                "token_id": int(token_id)
            })

        return {
            "success": True,
            "input": text,
            "tokens": tokens,
            "count": len(tokens)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_tokenize_breaks_text_into_tokens -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_command_handler.py
git commit -m "feat: implement tokenize command

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: CommandHandler - Generate Command

**Files:**
- Modify: `gpt_token_explorer.py` (CommandHandler)
- Modify: `tests/test_command_handler.py`

**Step 1: Write test for generate command**

Add to `tests/test_command_handler.py`:
```python
def test_generate_produces_tokens():
    """Generate command produces n tokens step-by-step"""
    manager = ModelManager(auto_load=True, model_name="HuggingFaceTB/SmolLM-135M")
    handler = CommandHandler(manager)

    result = handler.generate("Hello", n_tokens=3)

    assert result["success"] is True
    assert "steps" in result
    assert len(result["steps"]) == 3
    assert "final_text" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handler.py::test_generate_produces_tokens -v`
Expected: FAIL

**Step 3: Implement generate() method**

Add to `CommandHandler`:
```python
def generate(self, text, n_tokens=10, show_alternatives=3):
    """Generate n tokens autoregressively with probabilities

    Args:
        text: Input prompt
        n_tokens: Number of tokens to generate
        show_alternatives: Number of alternative tokens to show at each step

    Returns:
        dict with success, steps, and final_text
    """
    valid, error = self.validate_input(text)
    if not valid:
        return {"success": False, "error": error}

    try:
        input_ids = self.model_manager.tokenizer.encode(
            text,
            return_tensors='pt'
        )

        steps = []
        current_text = text

        for step in range(n_tokens):
            # Get next token probabilities
            with torch.no_grad():
                outputs = self.model_manager.model(input_ids)

            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Get top alternatives
            top_probs, top_ids = torch.topk(probs[0], show_alternatives)

            # Sample the most likely token
            next_token_id = top_ids[0]
            next_token = self.model_manager.tokenizer.decode([next_token_id])

            # Record alternatives
            alternatives = []
            for prob, token_id in zip(top_probs, top_ids):
                token = self.model_manager.tokenizer.decode([token_id])
                alternatives.append({
                    "token": token,
                    "probability": float(prob),
                    "selected": (token_id == next_token_id)
                })

            steps.append({
                "step": step + 1,
                "current_text": current_text,
                "alternatives": alternatives,
                "selected": next_token
            })

            # Append to input for next iteration
            current_text += next_token
            input_ids = torch.cat([
                input_ids,
                next_token_id.unsqueeze(0).unsqueeze(0)
            ], dim=1)

        return {
            "success": True,
            "prompt": text,
            "steps": steps,
            "final_text": current_text
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handler.py::test_generate_produces_tokens -v -s`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_command_handler.py
git commit -m "feat: implement generate command with step-by-step generation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: OutputFormatter - Basic Structure

**Files:**
- Modify: `gpt_token_explorer.py`
- Create: `tests/test_output_formatter.py`

**Step 1: Write test for OutputFormatter**

Create `tests/test_output_formatter.py`:
```python
import pytest
from gpt_token_explorer import OutputFormatter

def test_output_formatter_can_be_created():
    """OutputFormatter can be instantiated"""
    formatter = OutputFormatter()
    assert formatter is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_formatter.py::test_output_formatter_can_be_created -v`
Expected: ImportError

**Step 3: Add OutputFormatter class**

Add to `gpt_token_explorer.py`:
```python
class OutputFormatter:
    """Formats command results for terminal display"""

    # ANSI color codes
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    RESET = '\033[0m'

    @staticmethod
    def probability_bar(prob, max_width=20):
        """Create probability bar visualization"""
        filled = int(prob * max_width)
        return '█' * filled + '░' * (max_width - filled)

    @staticmethod
    def probability_color(prob):
        """Get color based on probability threshold"""
        if prob > 0.5:
            return OutputFormatter.GREEN
        elif prob > 0.1:
            return OutputFormatter.YELLOW
        else:
            return OutputFormatter.RED
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_output_formatter.py::test_output_formatter_can_be_created -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gpt_token_explorer.py tests/test_output_formatter.py
git commit -m "feat: add OutputFormatter with color codes and utilities

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: OutputFormatter - Format Methods

**Files:**
- Modify: `gpt_token_explorer.py` (OutputFormatter)

**Step 1: Implement format_complete()**

Add to `OutputFormatter`:
```python
@staticmethod
def format_complete(result):
    """Format complete command results"""
    if not result["success"]:
        return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

    lines = [f"\n{OutputFormatter.BOLD}Next Token Predictions:{OutputFormatter.RESET}\n"]

    for i, item in enumerate(result["results"], 1):
        prob = item["probability"]
        token = item["token"]
        token_id = item["token_id"]

        color = OutputFormatter.probability_color(prob)
        bar = OutputFormatter.probability_bar(prob)

        lines.append(
            f"   {i:2d}. {color}{token:20s}{OutputFormatter.RESET} "
            f"{prob:.4f} {bar} {OutputFormatter.GRAY}[{token_id}]{OutputFormatter.RESET}"
        )

    return "\n".join(lines) + "\n"

@staticmethod
def format_tokenize(result):
    """Format tokenize command results"""
    if not result["success"]:
        return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

    lines = [
        f"\n{OutputFormatter.BOLD}Tokenization:{OutputFormatter.RESET}",
        f"   Input: \"{result['input']}\"",
        f"   Tokens: {result['count']}\n"
    ]

    for i, token in enumerate(result["tokens"], 1):
        token_text = token["text"]
        token_id = token["token_id"]
        lines.append(
            f"   {i:2d}. {OutputFormatter.BLUE}\"{token_text}\"{OutputFormatter.RESET} "
            f"{OutputFormatter.GRAY}[ID: {token_id}]{OutputFormatter.RESET}"
        )

    lines.append(f"\n   {OutputFormatter.YELLOW}Note: Tokens split by Byte-Pair Encoding (BPE){OutputFormatter.RESET}\n")
    return "\n".join(lines)

@staticmethod
def format_generate(result):
    """Format generate command results"""
    if not result["success"]:
        return f"{OutputFormatter.RED}✗ {result['error']}{OutputFormatter.RESET}"

    lines = [f"\n{OutputFormatter.BOLD}Step-by-Step Generation:{OutputFormatter.RESET}\n"]

    for step_data in result["steps"]:
        step = step_data["step"]
        current = step_data["current_text"]
        selected = step_data["selected"]
        alternatives = step_data["alternatives"]

        lines.append(f"   {OutputFormatter.BOLD}Step {step}:{OutputFormatter.RESET} \"{current}\"")

        # Show top alternatives
        alt_strs = []
        for alt in alternatives[:3]:
            color = OutputFormatter.GREEN if alt["selected"] else OutputFormatter.GRAY
            alt_strs.append(f"{color}{alt['token']}{OutputFormatter.RESET} ({alt['probability']:.2f})")

        lines.append(f"   → {' | '.join(alt_strs)}\n")

    lines.append(f"   {OutputFormatter.BOLD}Final:{OutputFormatter.RESET} \"{result['final_text']}\"\n")
    return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add gpt_token_explorer.py
git commit -m "feat: implement OutputFormatter display methods

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: TokenREPL - Basic Structure

**Files:**
- Modify: `gpt_token_explorer.py`

**Step 1: Add TokenREPL class**

Add to `gpt_token_explorer.py`:
```python
class TokenREPL:
    """Interactive REPL for GPT token exploration"""

    COMMANDS = ['complete', 'generate', 'tokenize', 'help', 'quit', 'exit']

    def __init__(self, model_name=None):
        self.model_manager = None
        self.command_handler = None
        self.formatter = OutputFormatter()
        self.model_name = model_name

        # prompt_toolkit session
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

        # Show first-time tips
        self.print_tips()

        # Main loop
        while True:
            try:
                user_input = self.session.prompt('\ngpt> ').strip()

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
        """Print welcome banner"""
        print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}  GPT Token Explorer - Learn How LLMs Generate Text{OutputFormatter.RESET}")
        print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")

    def print_tips(self):
        """Print usage tips"""
        print(f"Try these commands:")
        print(f'  {OutputFormatter.BLUE}complete{OutputFormatter.RESET} "Hello, my name is"')
        print(f'  {OutputFormatter.BLUE}generate{OutputFormatter.RESET} "Once upon a time" 10')
        print(f'  {OutputFormatter.BLUE}tokenize{OutputFormatter.RESET} "ChatGPT is cool!"')
        print(f'  {OutputFormatter.BLUE}help{OutputFormatter.RESET}')

    def print_help(self):
        """Print help message"""
        help_text = f"""
{OutputFormatter.BOLD}Available Commands:{OutputFormatter.RESET}

  {OutputFormatter.BLUE}complete{OutputFormatter.RESET} <text>
      Show next token probabilities
      Example: complete "The capital of France is"

  {OutputFormatter.BLUE}generate{OutputFormatter.RESET} <text> [n]
      Generate n tokens step-by-step (default n=10)
      Example: generate "Once upon a time" 5

  {OutputFormatter.BLUE}tokenize{OutputFormatter.RESET} <text>
      Break text into tokens with IDs
      Example: tokenize "ChatGPT is amazing!"

  {OutputFormatter.BLUE}help{OutputFormatter.RESET}
      Show this help message

  {OutputFormatter.BLUE}quit{OutputFormatter.RESET} / {OutputFormatter.BLUE}exit{OutputFormatter.RESET}
      Exit the explorer
"""
        print(help_text)

    def execute_command(self, user_input):
        """Parse and execute user command"""
        parts = user_input.split(maxsplit=1)
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == 'complete':
            if not args:
                print(f"{OutputFormatter.RED}Usage: complete <text>{OutputFormatter.RESET}")
                return
            result = self.command_handler.complete(args)
            print(self.formatter.format_complete(result))

        elif cmd == 'tokenize':
            if not args:
                print(f"{OutputFormatter.RED}Usage: tokenize <text>{OutputFormatter.RESET}")
                return
            result = self.command_handler.tokenize(args)
            print(self.formatter.format_tokenize(result))

        elif cmd == 'generate':
            if not args:
                print(f"{OutputFormatter.RED}Usage: generate <text> [n]{OutputFormatter.RESET}")
                return

            # Parse args
            parts = args.rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit():
                text = parts[0]
                n = int(parts[1])
            else:
                text = args
                n = 10

            result = self.command_handler.generate(text, n_tokens=n)
            print(self.formatter.format_generate(result))

        else:
            print(f"{OutputFormatter.RED}Unknown command: {cmd}{OutputFormatter.RESET}")
            print("Type 'help' for available commands.")
```

**Step 2: Commit**

```bash
git add gpt_token_explorer.py
git commit -m "feat: implement TokenREPL with command execution

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Interactive Model Selection

**Files:**
- Modify: `gpt_token_explorer.py`

**Step 1: Add select_model_interactive() function**

Add before `main()`:
```python
def select_model_interactive():
    """Interactive model selector"""
    print(f"\n{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}  Select Language Model{OutputFormatter.RESET}")
    print(f"{OutputFormatter.BOLD}{'='*60}{OutputFormatter.RESET}\n")

    models = [
        {
            'name': 'HuggingFaceTB/SmolLM-135M',
            'display': 'SmolLM-135M (Recommended)',
            'size': '135MB',
            'load_time': '~3s',
            'description': 'Tiny but capable modern model',
            'pros': '✓ Fastest  ✓ Smallest download  ✓ Good for demos'
        },
        {
            'name': 'meta-llama/Llama-3.2-1B',
            'display': 'Llama 3.2 1B (Meta)',
            'size': '1GB',
            'load_time': '~15s',
            'description': 'Meta\'s smallest Llama 3.2 model',
            'pros': '✓ Best quality  ✓ Meta official  ✓ Modern architecture'
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
            choice = input("Choose model (1-2) [default: 1]: ").strip()
            if not choice:
                choice = '1'

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected = models[choice_idx]
                print(f"\n✓ Selected: {selected['display']}")
                return selected['name']
            else:
                print(f"{OutputFormatter.RED}Please enter 1 or 2{OutputFormatter.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{OutputFormatter.YELLOW}Using default (SmolLM-135M){OutputFormatter.RESET}")
            return models[0]['name']
        except EOFError:
            return models[0]['name']
```

**Step 2: Add main() function**

```python
def main():
    """Entry point for GPT token explorer"""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPT Token Explorer - Learn how LLMs generate text',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use (skips interactive selection)'
    )

    args = parser.parse_args()

    # If no model specified, show interactive selector
    model_name = args.model
    if model_name is None:
        model_name = select_model_interactive()

    repl = TokenREPL(model_name=model_name)
    repl.start()

if __name__ == '__main__':
    main()
```

**Step 3: Commit**

```bash
git add gpt_token_explorer.py
git commit -m "feat: add interactive model selection on startup

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: README Documentation

**Files:**
- Create: `README.md`

**Step 1: Write README**

Create `README.md`:
```markdown
# 🤖 GPT Token Explorer

Interactive Python REPL for learning how GPT models generate tokens. Educational tool demonstrating next-token prediction, probability distributions, and autoregressive generation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Next-Token Prediction** - See probability distributions for the next token
- **Step-by-Step Generation** - Watch autoregressive generation unfold
- **Tokenization** - Understand how text splits into BPE tokens
- **Model Comparison** - Compare SmolLM vs Llama predictions
- **Interactive Learning** - Rich REPL with colored output and progress bars

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd gpt-token-explorer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run - you'll choose a model interactively
./explore.sh

# Or specify model directly
./explore.sh --model HuggingFaceTB/SmolLM-135M
```

**First run:** Choose your model:
1. **SmolLM-135M** (recommended) - Fast, 135MB, ~3s load
2. **Llama 3.2 1B** (Meta) - Better quality, 1GB, ~15s load

## 💡 Usage

**See next-token probabilities:**
```
gpt> complete "The capital of France is"

   Next Token Predictions:
   1. Paris      0.8423 ████████████████
   2. the        0.0512 ██
   3. located    0.0234 █
```

**Generate text step-by-step:**
```
gpt> generate "Once upon a time" 5

   Step 1: "Once upon a time"
   → there (0.45) | was (0.32) | in (0.12)

   Step 2: "Once upon a time there"
   → was (0.78) | lived (0.11) | were (0.05)

   Final: "Once upon a time there was a"
```

**Understand tokenization:**
```
gpt> tokenize "ChatGPT is amazing!"

   Tokenization:
   Input: "ChatGPT is amazing!"
   Tokens: 5

   1. "Chat"    [ID: 13667]
   2. "GPT"     [ID: 38]
   3. " is"     [ID: 318]
   4. " amazing" [ID: 4998]
   5. "!"       [ID: 0]

   Note: Tokens split by Byte-Pair Encoding (BPE)
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `complete <text>` | Show next token probabilities | `complete "Hello world"` |
| `generate <text> [n]` | Generate n tokens step-by-step | `generate "Once upon a time" 10` |
| `tokenize <text>` | Break text into tokens | `tokenize "ChatGPT rocks!"` |
| `help` | Show all commands | |
| `quit` | Exit | |

## 🧠 How It Works

GPT models predict one token at a time through **autoregressive generation**:

1. **Input Processing** - Text converted to token IDs
2. **Forward Pass** - Model produces logits (scores) for each vocabulary token
3. **Softmax** - Logits converted to probabilities (sum to 1.0)
4. **Selection** - Highest probability token selected
5. **Repeat** - Selected token appended, process repeats

**Educational Focus:**
- See probability distributions visually
- Understand tokenization (BPE splitting)
- Watch step-by-step generation
- Compare different models

## 📋 Requirements

- Python 3.8+
- ~1GB disk space (model cache)
- ~2GB RAM for SmolLM, ~4GB for Llama

## 🎛️ Models

**SmolLM-135M** (default)
- Parameters: 135M
- Vocabulary: 49K tokens
- Load time: ~3s
- Best for: Quick demos, learning basics

**Llama 3.2 1B** (Meta)
- Parameters: 1B
- Vocabulary: 128K tokens
- Load time: ~15s
- Best for: Higher quality, Meta demonstrations

Models downloaded once to `~/.cache/huggingface/`

## 🤝 Contributing

Contributions welcome! This is an educational project.

## 📄 License

MIT License

---

**Educational tool** • Models from Hugging Face
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Add LICENSE

**Files:**
- Create: `LICENSE`

**Step 1: Create MIT LICENSE**

Create `LICENSE`:
```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 2: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 14: Final Integration Testing

**Files:**
- All existing files

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Manual testing with SmolLM**

Run: `./explore.sh`
Select: 1 (SmolLM)
Test all commands:
- `complete "The cat sat"`
- `generate "Hello" 5`
- `tokenize "Hello world"`
- `help`
- `quit`

Expected: All commands work, output is clear and educational

**Step 3: Manual testing with Llama (if accessible)**

Run: `./explore.sh --model meta-llama/Llama-3.2-1B`
Note: May require HF authentication
Test same commands

**Step 4: Commit**

```bash
git commit --allow-empty -m "test: verify all integration tests pass

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

✅ All tests pass
✅ Model loads in <30s (SmolLM)
✅ Commands execute quickly (<5s)
✅ Output is colored and clear
✅ Help text is educational
✅ README is comprehensive

---

**Total Tasks:** 14
**Estimated Time:** 3-4 hours
**Complexity:** Medium (similar to word2vec explorer)
