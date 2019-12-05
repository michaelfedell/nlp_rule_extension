import pandas as pd

reviews = pd.read_csv('../data/raw/reviews.csv', index_col=0)
limit = min(len(reviews), 10000)
reviews = reviews[:limit]

methods = ['naive', 'lemma', 'embedding', 'manual']
for method in methods:
    with open(f'../data/{method}_labeled_reviews.txt') as f:
        labels = [l.strip() for l in f.readlines()]  # get labels from each text file
    reviews[f'{method}_label'] = labels[:limit]

labels = reviews.drop(columns='document_text')
print(labels.apply(lambda x: x.value_counts()))
labels.to_csv('../data/all_labels.csv')
