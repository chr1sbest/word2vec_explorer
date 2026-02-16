# 🔤 Word2Vec Explorer

Interactive Python REPL for exploring word embeddings. Demonstrates the classic "king - man + woman = queen" analogy using Google's pre-trained word2vec model.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Analogy** - Vector arithmetic (king:man :: woman:?) → queen
- **Similarity** - Find semantically similar words
- **Distance** - Compute cosine similarity between words
- **Search** - Wildcard vocabulary search (`prog*` → programming, programmer...)
- **Inspect** - View raw 300-dimensional embeddings

Rich REPL with command history, autocomplete, and colored output.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/chr1sbest/word2vec_explorer.git
cd word2vec_explorer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run (downloads 1.5GB model on first run)
./explore.sh
```

## 💡 Usage

```
word2vec> analogy king:man woman:
   1. queen      0.7118 ████████████████
   2. monarch    0.6189 ████████████

word2vec> similar python 5
   1. pythons    0.6688 █████████████
   2. snake      0.6606 █████████████

word2vec> distance happy sad
   Similarity: 0.5355 █████████████████████

word2vec> find quantum*
   • quantum_mechanics
   • quantum_physics
   • quantum_computing

word2vec> vector king
   300 dimensions: [-0.32, 0.28, 0.15...]
   Stats: min=-0.64, max=0.61, mean=-0.03
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `analogy w1:w2 w3:` | Find X where w1:w2 :: w3:X | `analogy Paris:France Italy:` |
| `similar word [n]` | N most similar words (default 10) | `similar coffee 5` |
| `distance w1 w2` | Cosine similarity score | `distance cat dog` |
| `find pattern` | Search with wildcards | `find AI_*` |
| `vector word` | Show embedding | `vector king` |
| `help` | Show all commands | |
| `quit` | Exit | |

## 🧠 How It Works

Uses Google's word2vec model (300D vectors, 3M words, trained on 100B words from Google News).

**The "king - man + woman = queen" magic:**
- "king" vector = royalty + male
- Subtract "man" = remove male
- Add "woman" = add female
- Result closest to "queen" = royalty + female

## 📋 Requirements

- Python 3.8+
- ~2GB disk space for model

## 🔧 Troubleshooting

**Architecture mismatch error on Apple Silicon?**
```bash
# Delete and recreate the virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📚 References

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781) - Mikolov et al. (2013)
- [Gensim](https://radimrehurek.com/gensim/) - Model provider

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**Educational project** • Pre-trained model © Google Research (Apache 2.0)
