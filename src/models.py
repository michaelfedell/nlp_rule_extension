import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib

# Load labels for dataset
print('Loading data')
labels = pd.read_csv('../data/all_labels.csv', index_col='document_id')
X = joblib.load('../artifacts/doc_vectors.joblib')
X = X[:len(labels)]


def plot_conf_mat(conf_mat, class_names, label):
    dat = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    fig = plt.figure(figsize=(6, 6))
    plot = sns.heatmap(dat, annot=True, fmt='d', cbar=False)
    plot.yaxis.set_ticklabels(plot.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plot.xaxis.set_ticklabels(plot.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    plt.title(f'Confusion Matrix using {label}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


for label in labels:
    print(f'Training model on {label} data')
    y = labels[label]

    clf = LogisticRegression(solver='liblinear', random_state=903, multi_class='auto')
    clf.fit(X, y)
    print(clf.score(X, y))
    y_pred = clf.predict(X)
    joblib.dump(clf, f'../artifacts/{label}_model.joblib')

    class_names = clf.classes_
    conf_mat = confusion_matrix(y, y_pred, labels=class_names)
    fig = plot_conf_mat(conf_mat, class_names, label)
    fig.savefig(f'../artifacts/{label}_conf_mat.png')
