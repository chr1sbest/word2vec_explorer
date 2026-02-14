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
