import urllib.request
from itertools import permutations

import numpy as np

from ml.kmeans import KMeans

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def load_iris():
    raw = urllib.request.urlopen(IRIS_URL).read().decode("utf-8").strip()
    rows = [line.split(",") for line in raw.splitlines() if line.strip()]
    X = np.array([[float(v) for v in row[:4]] for row in rows])
    y = np.array([row[4] for row in rows])
    return X, y


def cluster_accuracy(y_true_str, y_pred, k):
    species = sorted(set(y_true_str))
    y_true = np.array([species.index(s) for s in y_true_str])
    best = max(
        permutations(range(k)),
        key=lambda p: np.mean([p[pred] == true for pred, true in zip(y_pred, y_true)]),
    )
    acc = np.mean([best[pred] == true for pred, true in zip(y_pred, y_true)])
    return acc, best


def main():
    X, y = load_iris()

    feature_names = [FEATURE_NAMES[i] for i in (2, 3)]
    X2 = X[:, (2, 3)]
    print(f"Dimensiones elegidas: {feature_names}")

    model = KMeans(k=3, max_iters=100, seed=42)
    labels = model.fit_predict(X2, track=True)

    acc, mapping = cluster_accuracy(y, labels, 3)
    print(f"Precisión (mejor mapeo clusters->especies) = {acc:.2%}")

    print("Clusters por especie:")
    for species in np.unique(y):
        counts = np.bincount(labels[y == species], minlength=3)
        print(f"  {species}: {counts.tolist()}")


if __name__ == "__main__":
    main()
