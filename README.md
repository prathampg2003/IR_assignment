
# Vector Space Model (VSM) for Information Retrieval

This repository contains a Python implementation of a **Vector Space Model (VSM)** for document retrieval. The system allows users to search a corpus of documents and rank the results based on cosine similarity. It leverages TF-IDF weighting and various text preprocessing techniques, such as tokenization, stop-word removal, lemmatization, and stemming.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Preprocessing and Indexing the Corpus](#1-preprocessing-and-indexing-the-corpus)
  - [2. Querying the Model](#2-querying-the-model)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Features

- **Text Preprocessing**: Includes tokenization, stop-word removal, lemmatization, and stemming.
- **TF-IDF Weighting**: Calculates term frequencies and applies inverse document frequency (IDF) for better relevance.
- **Cosine Similarity**: Ranks documents by their cosine similarity to the query vector using a **lnc.ltc** weighting scheme.
- **Efficient Ranking**: Retrieves the top 10 most relevant documents for a query.

## Requirements

To run the project, you need:

- **Python 3.7+**
- Required Python packages:
  - `nltk`
  - `os`, `re`, `collections`, `math` (Standard libraries)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vsm-information-retrieval.git
   cd vsm-information-retrieval
   ```

2. Install dependencies:
   ```bash
   pip install nltk
   ```

3. Download required `nltk` resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### 1. Preprocessing and Indexing the Corpus

First, initialize the `VectorSpaceModel` and index a directory containing your corpus of text files:

```python
from VectorSpaceModel import VectorSpaceModel

vsm = VectorSpaceModel()
vsm.process_corpus('path/to/corpus')
```

The `process_corpus()` method reads and preprocesses each document in the specified directory and builds an inverted index.

### 2. Querying the Model

Once the corpus is indexed, you can search for relevant documents by passing a query to `rank_documents()`:

```python
query = "search terms"
preprocessed_query = vsm.preprocess(query)
results = vsm.rank_documents(preprocessed_query)

# Output top-ranked documents
for doc, score in results:
    print(f"Document: {doc}, Score: {score}")
```

## How It Works

1. **Preprocessing**: Each document and query undergoes:
   - **Tokenization**: Splitting text into words.
   - **Stop-word Removal**: Filtering common English words (e.g., "and", "the").
   - **Lemmatization and Stemming**: Reducing words to their root form.

2. **Indexing**: An inverted index is built, mapping terms to documents they appear in, along with their term frequencies.

3. **Querying**: The system calculates the **TF-IDF** weights for query terms and compares them with the document vectors using **cosine similarity**.

4. **Ranking**: Documents are ranked based on the cosine similarity score and returned as a list of the top 10 results.

## Future Enhancements

- **Support Phrase and Proximity Search**: Allow users to search for exact phrases or terms within a certain proximity.
- **Handle Larger Corpora**: Implement more efficient data structures for larger datasets.
- **Explore Advanced Ranking Models**: Integrate ranking algorithms such as BM25.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
