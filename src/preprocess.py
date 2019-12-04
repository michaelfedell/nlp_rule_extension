import argparse
import csv
import multiprocessing as mp
import os
import typing as t
from pathlib import Path
import joblib
import numpy as np

from fastai.text import Tokenizer, SpacyTokenizer, Vocab
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = Tokenizer()
tok = SpacyTokenizer('en')


def tokenize(line: str) -> str:
    doc = ' '.join(tokenizer.process_text(line, tok))  # parse with FastAI tokenizer
    return doc + '\n'


def numericalize(doc: str, vocab: Vocab) -> t.List[int]:
    return vocab.numericalize(doc)


def build_vocab(docs: t.List[str], max_vocab: int = 10000, min_freq: int = 5) -> Vocab:
    return Vocab.create(docs, max_vocab=max_vocab, min_freq=min_freq)


def vectorize_all(docs: t.List[str],
                  wv_path: str = '../artifacts/w2v_embeddings.wv',
                  cores: int = 0) -> np.array:
    """

    :param docs: list of processed document strings (should already be lemmatized/tokenized)
    :param wv_path:
    :param cores:
    :return:
    """
    assert os.path.exists(wv_path)
    word_vec = KeyedVectors.load(wv_path)
    tfidf = TfidfVectorizer()
    tfidf.fit(docs)

    def vectorize(doc: str):
        doc_weight = 0  # total doc_weight captures length of doc
        # doc_vector will become tfidf-weighted average of w2v vectors
        doc_vector = np.zeros(word_vec.vector_size)
        lookup = tfidf.transform([doc])  # get tfidf scores for all words in this document
        for token in doc.split():
            vector = word_vec.word_vec(token)
            weight = lookup[0, tfidf.vocabulary_[token]]
            doc_vector += weight * vector
            doc_weight += weight
        return doc_vector / doc_weight

    cores = cores or mp.cpu_count() - 1
    print(f'Calculating tfidf-weighted W2V vectors for {len(docs)} documents')
    with mp.Pool(cores) as p:
        vectors = list(tqdm(p.imap(vectorize, docs), total=len(docs)))

    return vectors


if __name__ == '__main__':
    lines = []

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('--out_file', default='')
    args = parser.parse_args()

    data_path = Path(args.data_file)
    assert data_path.exists()

    # default to in_file + '_processed'
    out_path = Path(args.out_file or '../data/raw/processed_reviews.txt')
    print(f'Saving output to {out_path.absolute()}')

    print(f'Extracting text from {data_path.name}')
    with data_path.open() as f:
        reader = csv.DictReader(f)
        lines = [row['document_text'] for row in reader]

    print('\nTokenizing text')
    with mp.Pool(mp.cpu_count() - 1) as p:
        docs = list(tqdm(p.imap(tokenize, lines), total=len(lines)))

    docs = list(filter(None, docs))
    with out_path.open('w') as f:
        f.writelines(docs)

    # print('\nBuilding Vocab')
    # vocab = build_vocab(docs)
    # joblib.dump(vocab, '../artifacts/vocab.joblib')
    #
    # print('\nNumericalizing Documents')
    # def num_all(doc):
    #     """Wrapper function for multiprocessing call below"""
    #     return numericalize(doc, vocab)
    #
    # with mp.Pool(mp.cpu_count() - 1) as p:
    #     num_docs = list(tqdm(p.imap(num_all, docs), total=len(lines)))
    # joblib.dump(num_docs, '../artifacts/num_docs.joblib')

    doc_vectors = vectorize_all(docs)
    joblib.dump(doc_vectors, '../artifacts/doc_vectors.joblib')
