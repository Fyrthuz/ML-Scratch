import urllib.request

import numpy as np

from ml.kmeans import KMeans

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def load_iris():
    raw = urllib.request.urlopen(IRIS_URL).read().decode("utf-8").strip()
    rows = [line.split(",") for line in raw.splitlines() if line.strip()]
    X = np.array([[float(v) for v in row[:4]] for row in rows])
    y = np.array([row[4] for row in rows])
    return X, y


def main():
    X, y = load_iris()

    model = KMeans(k=3, max_iters=100, seed=42)
    labels = model.fit_predict(X, track=True)

    print("Etiquetas predichas:", labels)
    print("Clusters por especie:")
    for species in np.unique(y):
        counts = np.bincount(labels[y == species], minlength=3)
        print(f"  {species}: {counts.tolist()}")


if __name__ == "__main__":
    main()
