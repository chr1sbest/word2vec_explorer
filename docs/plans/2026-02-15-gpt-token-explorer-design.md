# GPT Token Explorer - Design Document

**Date:** 2026-02-15
**Status:** Approved
**Type:** Educational REPL Tool

## Overview

An interactive command-line tool for learning how GPT models generate tokens. Similar in design to the word2vec explorer, but focused on demonstrating next-token prediction, probability distributions, and autoregressive generation.

**Target Audience:** Developers learning about LLMs
**Educational Focus:** Next-token prediction, step-by-step generation, probability distributions, model confidence, comparing models
**Technical Depth:** Medium (basic attention visualization, no deep transformer internals)

## Architecture

### High-Level Structure

Single-file architecture following the proven word2vec explorer pattern:

```
gpt-token-explorer/
├── gpt_token_explorer.py    # Main file (~700 lines)
├── requirements.txt          # Dependencies
├── explore.sh               # Launch script
├── README.md                # Documentation
├── LICENSE                  # MIT
└── examples/                # Example outputs
    └── demo_outputs.md      # Sample interactions
```

### Four Main Classes

1. **ModelManager** (~150 lines)
   - Loads and manages HuggingFace transformer models
   - Handles model selection (SmolLM-135M, Llama 3.2 1B)
   - Manages tokenizer
   - Provides clean API for inference

2. **CommandHandler** (~200 lines)
   - Executes commands: complete, generate, tokenize, attention, compare
   - Validates input
   - Extracts probabilities from logits
   - Returns structured results

3. **OutputFormatter** (~150 lines)
   - Pretty-prints results with ANSI colors
   - Probability bars (like similarity bars in word2vec)
   - Token ID display
   - Attention heatmaps (text-based)

4. **TokenREPL** (~200 lines)
   - Interactive loop using prompt_toolkit
   - Command history and autocomplete
   - Help system
   - Model selection interface

### Dependencies

```
transformers>=4.30.0    # HuggingFace models
torch>=2.0.0           # Model inference
prompt_toolkit>=3.0.0  # Rich REPL interface
numpy>=1.21.0          # Array operations
tqdm>=4.62.0           # Progress bars
```

## Interactive Model Selection

On startup, users choose from 2 models:

```
============================================================
  Select Language Model
============================================================

1. SmolLM-135M (Recommended)
   Tiny but capable modern model
   Size: 135MB | Load time: ~3s
   ✓ Fastest  ✓ Smallest download  ✓ Good for demos

2. Llama 3.2 1B (Meta)
   Meta's smallest Llama 3.2 model
   Size: 1GB | Load time: ~15s
   ✓ Best quality  ✓ Meta official  ✓ Modern architecture

Choose model (1-2) [default: 1]: _
```

**Design Rationale:**
- SmolLM-135M: Fast demos, quick iteration, educational baseline
- Llama 3.2 1B: Shows Meta's latest, higher quality, model comparison

## Commands

### 1. `complete <text>`
Show next token probabilities.

**Input:** Text prompt
**Output:** Top-10 most likely next tokens with probabilities and bars

**Example:**
```
gpt> complete "The capital of France is"

   Next Token Predictions:
   1. Paris      0.8423 ████████████████
   2. the        0.0512 ██
   3. located    0.0234 █
   4. known      0.0156 █
   5. ...
```

**Implementation:**
```python
def complete(self, text, top_k=10):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    # Return formatted results
```

### 2. `generate <text> [n]`
Generate N tokens with step-by-step probabilities.

**Input:** Text prompt, number of tokens (default 10)
**Output:** Generated text showing probabilities at each step

**Example:**
```
gpt> generate "Once upon a time" 5

   Step 1: "Once upon a time"
   → there (0.45) | was (0.32) | in (0.12)

   Step 2: "Once upon a time there"
   → was (0.78) | lived (0.11) | were (0.05)

   [continues for 5 tokens]

   Final: "Once upon a time there was a"
```

**Implementation:**
- Autoregressive loop
- Show top-3 alternatives at each step
- Highlight selected token

### 3. `tokenize <text>`
Break text into tokens with IDs.

**Input:** Text string
**Output:** Token breakdown with IDs and subword explanations

**Example:**
```
gpt> tokenize "ChatGPT is amazing!"

   4 tokens:
   1. "Chat"    [ID: 13667]
   2. "GPT"     [ID: 38, 4503] (subword: G + PT)
   3. " is"     [ID: 318]
   4. " amazing" [ID: 4998]
   5. "!"       [ID: 0]

   Note: Words split by Byte-Pair Encoding (BPE)
```

**Educational Value:**
- Shows why some words need multiple tokens
- Explains BPE tokenization
- Reveals token ID structure

### 4. `attention <text>`
Show which previous tokens influenced predictions.

**Input:** Text prompt
**Output:** Text-based attention heatmap

**Example:**
```
gpt> attention "The cat sat on the mat"

   Attention weights (last layer, averaged across heads):

         The  cat  sat  on   the  mat
   The   1.0  0.3  0.1  0.0  0.2  0.0
   cat   0.4  1.0  0.6  0.2  0.1  0.0
   sat   0.1  0.5  1.0  0.4  0.2  0.1
   on    0.0  0.2  0.5  1.0  0.7  0.3
   the   0.1  0.1  0.3  0.6  1.0  0.5
   mat   0.0  0.2  0.2  0.3  0.6  1.0

   Darker = stronger influence
```

**Implementation:**
```python
def attention(self, text):
    outputs = model(input_ids, output_attentions=True)
    attention = outputs.attentions[-1]  # Last layer
    avg_attention = attention.mean(dim=1)[0]  # Average heads
    # Format as heatmap
```

### 5. `compare <text>`
Compare both models' predictions side-by-side.

**Input:** Text prompt
**Output:** Probability distributions from both models

**Example:**
```
gpt> compare "The meaning of life is"

   SmolLM-135M:                Llama 3.2 1B:
   1. to        0.234           1. 42        0.445
   2. 42        0.156           2. to        0.278
   3. the       0.098           3. a         0.087

   Difference: Llama assigns higher probability to "42"
```

**Educational Value:**
- Shows model differences
- Demonstrates training data effects
- Highlights uncertainty

### 6. `help`
Show all commands with examples.

### 7. `quit` / `exit`
Exit REPL.

## Data Flow

### Startup Flow
```
1. Display ASCII banner
2. Show interactive model selector
3. User chooses model (or Enter for default)
4. ModelManager.load_model()
   - Check HF cache (~/.cache/huggingface/)
   - Download if needed (progress bar + time estimate)
   - Load into memory (spinner)
   - Display: "✓ Ready! Model: SmolLM-135M, Vocab: 49,152"
5. Enter REPL loop
```

### Command Execution Flow
```
1. User enters command
2. Parse command and arguments
3. Validate input
4. CommandHandler.<method>() executes
5. Tokenize text
6. Run model inference
7. Extract logits/probabilities/attention
8. OutputFormatter formats results
9. Display colored output
10. Return to prompt
```

### Probability Extraction
```python
# Core algorithm for all commands
outputs = model(input_ids)
logits = outputs.logits[:, -1, :]  # Last token position
probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
top_k_probs, top_k_ids = torch.topk(probs, k=10)
```

## Error Handling

### Model Loading Failures
- **Download errors:** Show retry message, check internet connection
- **Authentication errors (Llama gated):** Explain HF token setup
  ```
  Llama 3.2 requires authentication:
  1. Go to https://huggingface.co/meta-llama/Llama-3.2-1B
  2. Accept terms
  3. Run: huggingface-cli login
  ```
- **Memory errors:** Suggest lighter model (SmolLM)

### Input Validation
- **Empty text:** Show usage example
- **Very long text (>512 tokens):** Truncate with warning
  ```
  ⚠ Input truncated to 512 tokens (model limit)
  ```
- **Invalid commands:** Suggest `help`

### Generation Edge Cases
- **Max tokens reached:** Show "(max reached)" indicator
- **EOS token produced:** Stop gracefully
- **Temperature=0:** Handle deterministic case

## User Experience Polish

### Startup Experience
```
============================================================
  GPT Token Explorer - Learn How LLMs Generate Text
============================================================

Loading SmolLM-135M...
⚙️ Initializing model ⠋ (~3 seconds)

✓ Ready! Vocabulary: 49,152 tokens

Try these commands:
  complete "Hello, my name is"
  generate "Once upon a time" 10
  tokenize "ChatGPT is cool!"
  help

gpt> _
```

### Output Formatting
- **High probability (>0.5):** Green text
- **Medium probability (0.1-0.5):** Yellow text
- **Low probability (<0.1):** Red text
- **Progress bars:** █ characters (like word2vec)
- **Token IDs:** Gray/dim text
- **Section headers:** Bold text

### Performance
- Cache model and tokenizer in memory
- No re-loading between commands
- Fast inference:
  - SmolLM-135M: <1s per command
  - Llama 3.2 1B: <3s per command

## Testing Strategy

### Unit Tests
- `test_model_manager.py`: Model loading, tokenization
- `test_command_handler.py`: Probability extraction, generation logic
- `test_output_formatter.py`: Color codes, formatting

### Integration Tests
- Test full command flow
- Verify probability distributions sum to 1
- Check attention weights shape

### Manual Testing
- Test with both models
- Verify educational explanations are clear
- Check error messages are helpful

## Technical Specifications

### Model Details

**SmolLM-135M:**
- Parameters: 135 million
- Vocabulary: 49,152 tokens
- Context length: 2048 tokens
- Architecture: Llama-like transformer
- Training: FineWeb, DCLM datasets

**Llama 3.2 1B:**
- Parameters: 1 billion
- Vocabulary: 128,000 tokens
- Context length: 8192 tokens (we'll use 2048 for consistency)
- Architecture: Llama 3.2 transformer
- Training: Meta's proprietary dataset

### File Size Estimates
- `gpt_token_explorer.py`: ~700 lines / ~25KB
- `requirements.txt`: ~100 bytes
- `README.md`: ~5KB
- Total code: <30KB
- Models downloaded separately to HF cache

## Success Criteria

✅ Users understand next-token prediction
✅ Users see probability distributions clearly
✅ Users understand autoregressive generation
✅ Users can compare model behaviors
✅ Tool loads in <30s (SmolLM) or <60s (Llama)
✅ Commands execute in <5s
✅ Clear, educational output formatting

## Future Enhancements (Out of Scope)

- Temperature slider
- Beam search visualization
- Multi-layer attention analysis
- Export results to JSON
- Web interface
- Additional models

---

**Approved:** 2026-02-15
**Next Step:** Create implementation plan with writing-plans skill
