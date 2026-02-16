#!/bin/bash
# Word2Vec Explorer Launcher
cd "$(dirname "$0")"
source venv/bin/activate
python3 word2vec_explorer.py "$@"
