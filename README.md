# Probabilistic Record Matching

A robust and explainable system for matching records based on PII (Personally Identifiable Information) using language model embeddings and machine learning.

## Overview

This project implements a probabilistic approach to record linkage that:
- Uses **semantic embeddings** to capture fuzzy similarities between text fields
- Employs **Logistic Regression** to learn optimal field weights automatically
- Handles **missing data** gracefully with a conservative masking strategy
- Provides **full explainability** through interpretable feature weights

## Key Features

- **Column-wise Similarity**: Each field (Name, Address, City, Date of Birth) is embedded independently using `all-MiniLM-L6-v2`
- **Learnable Importance**: The model learns which fields matter most (e.g., Name > Date of Birth)
- **Missing Data Handling**: Sets similarity to 0 for missing fields to avoid false positives
- **Comprehensive Evaluation**: Includes precision, recall, F1 score, and error analysis
- **Interactive Demo**: Step-by-step Jupyter notebook with visualizations

## Project Structure

```
.
├── data_generator.py                    # Synthetic PII data generation with controlled errors
├── probabilistic_matching_demo.ipynb    # Interactive walkthrough notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Interactive Notebook (Recommended)

Run the Jupyter notebook for a guided walkthrough:

```bash
uv run jupyter notebook probabilistic_matching_demo.ipynb
```

The notebook includes:
1. Data generation with synthetic PII
2. Similarity computation using embeddings
3. Model training with Logistic Regression
4. Match prediction with configurable threshold (default: 0.7)
5. Side-by-side comparison of matched records
6. Accuracy metrics and error analysis
7. Probability distribution visualization

## How It Works

### 1. Data Generation
Synthetic PII records are generated with:
- Realistic names, addresses, cities, and dates of birth
- Controlled perturbations (typos, character swaps)
- Random missing data (~10% per field)

### 2. Similarity Computation
For each field independently:
- Text is encoded into dense vectors using a sentence transformer model
- Cosine similarity is computed between all pairs
- Missing values are masked (similarity = 0)

### 3. Model Training
- Positive pairs: Records with the same `original_id`
- Negative pairs: Randomly sampled non-matching records (5x positives)
- Features: Similarity scores for each field
- Model: Logistic Regression with balanced class weights

### 4. Prediction
- Pairs with probability > 0.7 are classified as matches
- Trivial exact matches are filtered out
- Results include confidence scores and ground truth labels

## Performance

The system achieves high precision by using a conservative threshold (0.7), which reduces false positives at the cost of potentially missing some true matches (lower recall). Performance metrics are displayed in the notebook.

## Design Rationale

### Why Column-wise Embeddings?
- **Explainability**: You can see which fields drove the match decision
- **Robustness**: Missing data in one field doesn't corrupt the entire representation
- **Learnable Weights**: The model discovers that Name matters more than Gender

### Why Not Concatenate All Fields?
- Structural changes (missing fields) would shift the entire embedding
- Loss of interpretability
- Harder to debug false positives/negatives

### Why Logistic Regression?
- Probabilistic output (match confidence)
- Interpretable coefficients (field importance)
- Fast training and inference
- Sufficient for this structured data problem

## Dependencies

- `sentence-transformers`: Semantic text embeddings
- `scikit-learn`: Logistic Regression and similarity metrics
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `faker`: Synthetic data generation

## License

This is a pilot/demo project for educational and evaluation purposes.
