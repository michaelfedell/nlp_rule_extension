import sys
from collections import Counter
from pathlib import Path
import multiprocessing as mp

import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# Simple rules mapping category to list of keywords
RULES = {
    'product': ['beautiful', '', ''],
    'build_quality': ['broken', 'bent', 'rip'],
    'pricing': ['cost', 'cheap', 'expensive'],
    'order_fulfillment': ['deliver', 'late', 'ship'],
    'customer_service': ['service', 'email', 'refund'],
    'registry': ['wedding', 'shower', 'register'],
    'holiday_seasonal': ['season', 'christmas', 'summer']
}

# Can specify priority in case of ties?
PRIORITY = [
    'customer_service',
    'order_fulfillment',
    'pricing',
    'build_quality',
    'registry',
    'holiday_seasonal',
    'product'
]

# Invert rules so that keywords map to a category
cat_map = {l: k for k, v in RULES.items() for l in v}


def classify_naive(doc: str) -> str:
    """Classify a document by direct matching of keywords for each category"""
    doc = doc.lower().split()
    keys = [cat_map.get(w, '') for w in doc]
    keys = list(filter(None, keys))
    count = Counter(keys)
    top = count.most_common(1)
    return top[0][0] if top else 'unk'  # return key with highest count


def classify_lemma(doc: str) -> str:
    """Classify a document by keyword frequency but examining word lemmas for more general match"""
    doc = nlp(doc)
    lemma = [t.lemma_ for t in doc]
    keys = [cat_map.get(w, '') for w in lemma]
    keys = list(filter(None, keys))
    count = Counter(keys)
    top = count.most_common(1)
    return top[0][0] if top else 'unk'  # return key with highest count


def classify_embedded(doc: str) -> str:
    """Classify a document with the expanded embedding vectors for category keywords"""
    # TODO: implement
    return 'unk'


if __name__ == '__main__':
    assert len(sys.argv) == 2
    review_path = Path(sys.argv[1])  # Path to raw reviews file
    assert review_path.exists()
    with review_path.open() as f:
        header = f.readline()
        reviews = [l.split(',')[1] for l in f.readlines()]

    methods = {
        'naive': classify_naive,
        'lemma': classify_lemma,
        'embedded': classify_embedded
    }

    for method in methods:
        print('Producing labels via', method, 'strategy')
        with mp.Pool(mp.cpu_count() - 1) as p:
            labels = list(tqdm(p.imap(methods[method], reviews), total=len(reviews)))
        with open(f'data/{method}/labeled_reviews.csv') as f:
            f.writelines(','.join(l) + '\n' for l in zip(labels, reviews))
