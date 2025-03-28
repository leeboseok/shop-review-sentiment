import numpy as np
from app.ml.model_loader import clf, vectorizer

def classify(document: str):
    label = {0: 'negative', 1: 'positive'}
    X = vectorizer.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document: str, y: int):
    X = vectorizer.transform([document])
    clf.best_estimator_.partial_fit(X, [y])
