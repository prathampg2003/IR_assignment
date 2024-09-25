import os
import math
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = {}  # Stores term -> (df, [(doc_id, term_freq)])
        self.doc_lengths = {}  # Stores doc_id -> document length
        self.doc_count = 0     # Number of documents in the corpus
        self.doc_id_to_file = {}  # Maps doc_id to file name
        self.stop_words = set(stopwords.words('english'))  # Stop words set
        self.lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
        self.stemmer = PorterStemmer()  # Initialize stemmer

    def preprocess(self, text):
        """
        Tokenizes, removes stop words, lemmatizes, and stems the text.
        """
        # Tokenize and remove non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization and stemming (optional: you can choose one)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        stemmed_tokens = [self.stemmer.stem(word) for word in lemmatized_tokens]

        return stemmed_tokens

    def add_document(self, doc_id, terms):
        """
        Adds a document's terms to the index.
        """
        term_freqs = defaultdict(int)
        for term in terms:
            term_freqs[term] += 1

        for term, tf in term_freqs.items():
            if term not in self.dictionary:
                self.dictionary[term] = (0, [])
            df, postings = self.dictionary[term]
            postings.append((doc_id, tf))
            self.dictionary[term] = (df + 1, postings)

        # Calculate and store document length (used for normalization)
        doc_length = math.sqrt(sum((1 + math.log10(tf))**2 for tf in term_freqs.values()))
        self.doc_lengths[doc_id] = doc_length
        self.doc_count += 1

    def tf_idf(self, term, doc_id, tf, for_query=False):
        """
        Computes the tf-idf value for a term in a document or query.
        """
        df, _ = self.dictionary[term]
        idf = math.log10(self.doc_count / df)
        log_tf = 1 + math.log10(tf)

        if for_query:
            return log_tf * idf  # Use idf for query
        return log_tf  # Don't use idf for documents

    def rank_documents(self, query_terms):
        """
        Ranks documents by cosine similarity between the query and each document.
        """
        query_term_freqs = defaultdict(int)
        for term in query_terms:
            if term in self.dictionary:
                query_term_freqs[term] += 1

        query_vector = {}
        for term, tf in query_term_freqs.items():
            query_vector[term] = self.tf_idf(term, None, tf, for_query=True)

        # Normalize query vector
        query_length = math.sqrt(sum(weight**2 for weight in query_vector.values()))
        if query_length > 0:
            query_vector = {term: weight / query_length for term, weight in query_vector.items()}

        # Calculate cosine similarity for each document
        scores = defaultdict(float)
        for term, query_weight in query_vector.items():
            df, postings = self.dictionary.get(term, (0, []))
            for doc_id, tf in postings:
                doc_weight = self.tf_idf(term, doc_id, tf)
                scores[doc_id] += query_weight * doc_weight

        # Normalize document scores by their lengths
        for doc_id in scores:
            if self.doc_lengths[doc_id] > 0:
                scores[doc_id] /= self.doc_lengths[doc_id]

        # Sort by scores (descending), then by doc_id (ascending for ties)
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        # Convert doc_id to filenames and return result
        ranked_filenames = [(self.doc_id_to_file[doc_id], score) for doc_id, score in ranked_docs[:10]]

        return ranked_filenames

    def process_corpus(self, corpus_dir):
        """
        Traverse all text files in the corpus directory and index them.
        """
        for doc_id, filename in enumerate(os.listdir(corpus_dir), start=1):
            if filename.endswith(".txt"):
                with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    terms = self.preprocess(text)
                    self.add_document(doc_id, terms)
                    self.doc_id_to_file[doc_id] = filename
        print(f"Processed {self.doc_count} documents.")

# Example usage:

# Initialize VSM
vsm = VectorSpaceModel()

# Path to the corpus directory (adjust path as needed)
corpus_directory = "Corpus"

# Process the corpus (traverse all text files)
vsm.process_corpus(corpus_directory)

# Search with a free-text query
query = "Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation"
query_terms = vsm.preprocess(query)
result = vsm.rank_documents(query_terms)

# Print the top relevant documents with scores
print(f"Top relevant documents for query '{query}': {result}")
