#!/usr/bin/env python3
"""
Generate 2D visualization of word embeddings to demonstrate how vector space
captures semantic relationships.

This creates a plot showing how related words cluster together in vector space.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

def visualize_word_relationships():
    """Create 2D visualization showing word relationships"""

    print("Loading FastText model...")
    model = api.load('fasttext-wiki-news-subwords-300')

    # Define word groups with semantic relationships
    # Using fewer words to avoid overlap, selected for good separation
    word_groups = {
        'Royalty': ['king', 'queen'],
        'Family': ['uncle', 'aunt', 'father'],
        'Food': ['sushi', 'pizza', 'taco'],
        'Countries': ['japan', 'italy', 'spain', 'china']
    }

    # Flatten all words and get their vectors
    all_words = []
    group_labels = []
    for group, words in word_groups.items():
        all_words.extend(words)
        group_labels.extend([group] * len(words))

    print(f"Getting vectors for {len(all_words)} words...")
    vectors = np.array([model[word] for word in all_words])

    # Reduce to 2D using PCA
    print("Reducing 300D vectors to 2D...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color scheme for each group
    colors = {
        'Royalty': '#e74c3c',      # red
        'Family': '#3498db',       # blue
        'Food': '#2ecc71',         # green
        'Countries': '#f39c12'     # orange
    }

    # Plot points by group
    for group in word_groups.keys():
        mask = [label == group for label in group_labels]
        group_coords = coords_2d[mask]
        ax.scatter(group_coords[:, 0], group_coords[:, 1],
                  s=200, alpha=0.6, c=colors[group], label=group,
                  edgecolors='black', linewidth=1.5)

    # Add word labels with better positioning
    for i, word in enumerate(all_words):
        ax.annotate(word,
                   xy=(coords_2d[i, 0], coords_2d[i, 1]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=13,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='white',
                            alpha=0.9,
                            edgecolor=colors[group_labels[i]],
                            linewidth=2))

    # Styling
    ax.set_title('Word2Vec: How Words Cluster in Vector Space\n' +
                '300-dimensional embeddings reduced to 2D using PCA',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    # Add explanation text
    explanation = (
        "Semantic relationships in vector space:\n"
        "• Royalty: king, queen\n"
        "• Family: uncle, aunt, father\n"
        "• Food: sushi, pizza, taco\n"
        "• Countries: japan, italy, spain, china"
    )
    ax.text(0.02, 0.02, explanation,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = 'examples/word_embedding_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    print(f"   Size: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.1%} of variance captured in 2D")

if __name__ == '__main__':
    visualize_word_relationships()
