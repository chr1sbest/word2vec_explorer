# Word2Vec Explorer - Design Document

**Date:** 2026-02-14
**Status:** Approved

## Overview

An educational interactive Python REPL for exploring word2vec embeddings through vector arithmetic and similarity operations. Uses pre-trained Google News word2vec model with a rich command-line interface.

## Goals

- Enable hands-on exploration of word embeddings
- Demonstrate classic word2vec operations (king - man + woman = queen)
- Provide immediate feedback for learning vector semantics
- Keep setup simple with minimal dependencies

## Architecture

### Three-Layer Design

**1. Model Layer**
- Loads pre-trained Word2Vec model using gensim
- Uses gensim's downloader API for model acquisition
- Loads on startup with progress indicator (30-60 second one-time cost)
- Provides clean API wrapping gensim's KeyedVectors

**2. Command Layer**
- Implements five core operations:
  - `analogy word1 word2 word3` - finds X where "word1:word2 :: word3:X"
    - Uses the classic word2vec formula: word1 - word2 + word3
    - Example: king - man + woman = queen
  - `similar word [n]` - finds N most similar words (default 10)
  - `distance word1 word2` - computes cosine similarity score
  - `find pattern` - searches vocabulary with wildcard support
  - `vector word` - displays the full embedding vector

**3. REPL Layer**
- Interactive interface built with prompt_toolkit
- Features: command history, autocompletion, syntax highlighting
- Formatted output with colors and aligned columns
- Contextual help and error messages

### Dependencies

- `gensim` - model loading and vector operations
- `numpy` - vector math (bundled with gensim)
- `prompt_toolkit` - rich REPL experience

## Components

### ModelManager
- Handles model loading and caching
- Provides vocabulary access and validation
- Wraps gensim KeyedVectors operations

### CommandHandler
- Parses and validates commands
- Implements the five operations
- Returns structured results

### OutputFormatter
- Pretty-prints results with colors
- Aligns columns for readability
- Formats vectors for display

### REPL
- prompt_toolkit session management
- Command parsing and dispatch
- Error handling and user feedback

## Data Flow

```
User Input → Parse Command → Validate Words → Call Gensim API → Format Results → Display
```

## Error Handling

**Word Not in Vocabulary:**
- Show clear error message
- Suggest similar words using fuzzy matching
- Use `find` command to show close matches

**Invalid Syntax:**
- Display usage examples for the command
- Show list of available commands

**Model Loading Failures:**
- Clear error message with download instructions
- Fallback guidance for manual model download

## Testing Strategy

As an educational tool (not production software):
- Manual testing of core operations
- Example queries in README for validation
- User testing for UX feedback

## File Structure

```
word2vec_explorer.py    # Main script with all components
docs/
  plans/
    2026-02-14-word2vec-explorer-design.md  # This file
README.md               # Setup instructions and examples
requirements.txt        # Dependencies
```

## Success Criteria

- Model loads successfully on first run
- All five commands work correctly
- Clear error messages for edge cases
- Pleasant REPL experience with history and colors
- README includes compelling examples

## Future Enhancements (Out of Scope)

- Model training capability
- Visualization (PCA/t-SNE plots)
- Batch query processing
- Export results to file
- Custom model support
