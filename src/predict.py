import os
import sys

import joblib
from gensim.models import KeyedVectors

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from embeddings import vectorize
from preprocess import tokenize
from rules import METHODS

METHODS['manual'] = lambda x: 'NA'
word_vec = KeyedVectors.load('../artifacts/w2v_embeddings.wv')
tfidf = joblib.load('../artifacts/tfidf.joblib')
models = {m: joblib.load(f'../artifacts/{m}_label_model.joblib') for m in METHODS}


def predict(doc):
    doc = tokenize(doc).rstrip()
    doc_vector = vectorize(doc, word_vec, tfidf)
    predictions = {}
    for method in METHODS:
        predictions[method] = {
            'rule': METHODS[method](doc),
            'model': models[method].predict([doc_vector])[0]
        }

    return predictions
