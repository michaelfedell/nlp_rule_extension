import pandas as pd

reviews = pd.read_csv('../data/raw/reviews.csv', index_col=0)

methods = ['naive', 'lemma', 'embedded']
for method in methods:
    with open(f'../data/{method}/labeled_reviews.csv') as f:
        labels = [r.split(',')[0] for r in f]  # just get label from each file, not review text
        reviews[f'{method}_label'] = labels[:len(reviews)]

labels = reviews.drop(columns='document_text')
print(labels.apply(lambda x: x.value_counts()))
