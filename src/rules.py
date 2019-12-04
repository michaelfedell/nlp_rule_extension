import multiprocessing as mp
import os
import sys
import typing as t
from collections import Counter, defaultdict
from pathlib import Path
import re

import spacy
from gensim.models import KeyedVectors
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# Simple rules mapping category to list of keywords
RULES = {
    'product': ['beautiful', 'perfect'],
    'build_quality': ['broken', 'bent', 'quality'],
    'pricing': ['cost', 'cheap', 'expensive'],
    'order_fulfillment': ['deliver', 'pack', 'ship'],
    'customer_service': ['service', 'email', 'refund'],
    'registry': ['wedding', 'shower', 'register'],
    'holiday_seasonal': ['season', 'holiday', 'summer']
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


def w2v_neighbors(words: t.List[str],
                  wv_path: str = '../artifacts/w2v_embeddings.wv',
                  k: int = 7) -> t.Dict[str, t.List[str]]:
    """Find the top k neighbors for each word in a list using a saved Word2Vec model"""
    assert os.path.exists(wv_path)
    word_vec = KeyedVectors.load(wv_path)

    word_map = defaultdict(list)
    for word in words:
        try:
            neighbors = [tup[0] for tup in word_vec.most_similar(positive=[word], topn=k)]
        except KeyError:
            print(f'Keyword "{word}" was not found in WV vocab and thus will not be expanded')
            neighbors = []

        word_map[word] = neighbors

    return word_map


def expand_rules(neighbors: t.Dict[str, t.List[str]]):
    """
    Expand the set of keywords for each category based on word neighbors

    :param neighbors: maps each keyword in RULES to list of neighbors
    :return: new cat_map with expanded entries
    """

    cat_map = {}
    dupes = []
    for cat in RULES:
        for kw in RULES[cat]:  # kw is each keyword for each category in original ruleset
            cat_map[kw] = cat  # ensure original word is present in expanded mapping

            # Map the keyword and all of its neighbors to the current category
            for w in neighbors[kw]:
                if cat_map.get(w) is not None and cat_map.get(w) != cat:
                    print(f'WARNING: expanded kw: {w} found in {cat} but '
                          f'already exists in {cat_map.get(w)}... To avoid ambiguity, '
                          f'this word will be removed from mapping entirely')
                    dupes.append(w)  # remember that this word is ambiguous - remove once done
                cat_map[w] = cat
    for dupe in set(dupes):
        del cat_map[dupe]

    return cat_map


# Invert rules so that keywords map to a category
default_cat_map = {kw: c for c, r in RULES.items() for kw in r}
# Re-key cat_map with keyword lemmas
lemma_cat_map = {nlp(kw)[0].lemma_: cat for kw, cat in default_cat_map.items()}
# flatten all keywords and add keyword neighbors to cat_map
all_kws = [w for cat in RULES for w in RULES[cat]]
expanded_cat_map = expand_rules(w2v_neighbors(all_kws))


def classify(doc: str, use_lemma: bool = False, expand_cat_map: bool = False) -> str:
    """
    Classify a document by matching its words to a mapping of keywords to category

    :param doc: text to be categorized according to cat_map
    :param use_lemma: will use a lemmatizer to simplify words in doc if true
    :param expand_cat_map: expand the default_cat_map to include embedded neighbors if true
    :return: string corresponding to category of document (one of keys in RULES)
    """
    assert not (use_lemma and expand_cat_map)  # cannot currently combine these strategies
    # use kw neighbors to expand category mapping
    cat_map = expanded_cat_map if expand_cat_map else default_cat_map
    # split doc into tokens
    if use_lemma:
        # Use spacy lemmatizer to simplify words in both doc and map
        tokens = [tok.lemma_ for tok in nlp(doc)]
        cat_map = lemma_cat_map
    else:
        # lowercase, remove punctuation, and split words
        non_alpha = re.compile(r'[^a-z]')
        tokens = [non_alpha.sub('', tok.strip()) for tok in doc.lower().split()]

    keys = [cat_map.get(w, '') for w in tokens]
    keys = list(filter(None, keys))

    count = Counter(keys)
    top = count.most_common(1)
    return top[0][0] if top else 'unk'  # return key with highest count


def classify_naive(doc: str) -> str:
    """Classify a document by direct matching of keywords for each category"""
    return classify(doc, use_lemma=False, expand_cat_map=False)


def classify_lemma(doc: str) -> str:
    """Classify a document by keyword frequency, examining word lemmas for more general match"""
    return classify(doc, use_lemma=True, expand_cat_map=False)


def classify_embedded(doc: str) -> str:
    """Classify a document by keyword frequency including neighbors mined from embedding space"""
    return classify(doc, use_lemma=False, expand_cat_map=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    review_path = Path(sys.argv[1])  # Path to raw reviews file
    assert review_path.exists()
    with review_path.open() as f:
        header = f.readline()
        # First field is document_id; text may contain extra commas or quotation marks
        reviews = [','.join(l.split(',')[1:]).replace('"', '') for l in f.readlines()]

    methods = {
        'naive': classify_naive,
        'lemma': classify_lemma,
        'embedded': classify_embedded
    }

    for method in methods:
        print('Producing labels via', method, 'strategy')
        with mp.Pool(mp.cpu_count() - 1) as p:
            labels = list(tqdm(p.imap(methods[method], reviews), total=len(reviews)))
        with open(f'../data/{method}/labeled_{review_path.name}', 'w', encoding='utf-8') as f:
            f.writelines(','.join(l) for l in zip(labels, reviews))
