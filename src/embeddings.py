import multiprocessing as mp
import typing as t
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

CORPUS = Path('../artifacts/processed_reviews.txt')


def vectorize(doc: str, word_vec, tfidf):
    doc_weight = 0  # total doc_weight captures length of doc
    # doc_vector will become tfidf-weighted average of w2v vectors
    doc_vector = np.zeros(word_vec.vector_size)
    lookup = tfidf.transform([doc])  # get tfidf scores for all words in this document
    for token in doc.split():
        try:
            vector = word_vec.word_vec(token)
            weight = lookup[0, tfidf.vocabulary_[token]]
        except KeyError:
            vector = np.zeros(word_vec.vector_size)
            weight = 0
        doc_vector += weight * vector
        doc_weight += weight
    if doc_weight == 0:
        return np.zeros(word_vec.vector_size)
    return doc_vector / doc_weight


def vectorize_all(docs: t.List[str],
                  word_vec: KeyedVectors,
                  cores: int = 0) -> np.array:
    """

    :param docs: list of processed document strings (should already be lemmatized/tokenized)
    :param wv_path:
    :param cores:
    :return:
    """
    tfidf = TfidfVectorizer()
    tfidf.fit(docs)
    joblib.dump(tfidf, '../artifacts/tfidf.joblib')

    print(f'Calculating tfidf-weighted W2V vectors for {len(docs)} documents')
    # TODO: currently bork due to unpicklable functions (multiprocessing does not
    #  play nice with local functions)
    # cores = cores or mp.cpu_count() - 1
    # with mp.Pool(cores) as p:
    #     vectors = list(tqdm(p.imap(vectorize, docs), total=len(docs)))
    vectors = []
    for doc in tqdm(docs):
        vectors.append(vectorize(doc, word_vec, tfidf))

    return vectors


if __name__ == '__main__':
    w2v = Word2Vec(
        corpus_file=str(CORPUS),
        size=300
    )
    word_vec = w2v.wv
    word_vec.save('../artifacts/w2v_embeddings.wv')

    with CORPUS.open() as f:
        docs = [l.rstrip() for l in f]

    doc_vectors = vectorize_all(docs, word_vec)
    joblib.dump(doc_vectors, '../artifacts/doc_vectors.joblib')
