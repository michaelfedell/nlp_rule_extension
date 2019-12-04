from pathlib import Path

from gensim.models import Word2Vec

CORPUS = Path('../data/raw/processed_reviews.txt')

if __name__ == '__main__':
    w2v = Word2Vec(
        corpus_file=str(CORPUS),
        size=300
    )
    word_vec = w2v.wv
    word_vec.save('../artifacts/w2v_embeddings.wv')
