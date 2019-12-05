import csv
import multiprocessing as mp
import typing as t
from pathlib import Path

import joblib
from fastai.text import Tokenizer, SpacyTokenizer, Vocab
from tqdm import tqdm

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


if __name__ == '__main__':
    data_path = Path('../data/raw/reviews.csv')
    assert data_path.exists()

    # default to in_file + '_processed'
    out_path = Path('../artifacts/processed_reviews.txt')
    print(f'Saving output to {out_path.absolute()}')

    print(f'Extracting text from {data_path.name}')
    with data_path.open() as f:
        reader = csv.DictReader(f)
        lines = [row['document_text'] for row in reader]

    print('\nTokenizing text')
    with mp.Pool(mp.cpu_count() - 1) as p:
        docs = list(tqdm(p.imap(tokenize, lines), total=len(lines)))

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
