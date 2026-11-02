import os

import numpy as np


class KMeans():

    def __init__(self, k, max_iters=300, tol=0.0001, dist_metric='euclidean', seed = None):
        self.n_centroids = k
        self.max_iters = max_iters
        self.tol = tol
        self.dist_metric = dist_metric
        self.centroids = None
        np.random.default_rng(seed)

    def _distance_to_centroids(self, points):
        try:
            if self.dist_metric == 'euclidean':
                # (N, D), (C, D) --> (N, C)  === (N,1,D) - (1, C, D) --> (N,C)
                # squared_diff (N, C, D)
                squared_diff = (points[:, np.newaxis, :] - self.centroids[np.newaxis, :, :])**2
                # dists (N, C)
                dists = np.sqrt(np.sum(squared_diff, axis=-1))
                return dists
            else:
                raise Exception()

        except Exception as e:
            raise("Métrica de distancia para KMeans no válida")

    def _get_label(self, points_dist):
        return np.argmin(points_dist, axis=1)

    def _init_centroids(self, n_features, min, max):
        self.centroids = np.random.random((self.n_centroids, n_features))
        self.centroids = min + self.centroids * (max-min)

    def _update_centroids(self, X, labels):
        # X = (N, D), Labels = np.array(N), Centroides = (K, D)
        # (N, K)
        ohe_labels = np.eye(self.n_centroids)[labels]

        # K, D = (K, N) @ (N, D)
        suma_valores = ohe_labels.T @ X

        # (K,)
        counts = np.maximum(ohe_labels.sum(axis=0).reshape((-1, 1)), 1)

        self.centroids = suma_valores / counts

        return self.centroids

    def _make_snapshot(self, X, labels, centroids, figsize=(6, 4)):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25, alpha=0.6)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=range(self.n_centroids),
            cmap="viridis",
            s=200,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
        )
        return fig, ax, plt

    def _track_frame(self, X, labels, centroids, X_plot, out_dir, n_iter):
        import matplotlib.pyplot as plt

        fig, ax, plt_ref = self._make_snapshot(X, labels, centroids)
        if X_plot is not None:
            ax.set_title(
                f"Iteración {n_iter} (mostrando 2 primeras dims de {X.shape[1]})"
            )
        else:
            ax.set_title(f"Iteración {n_iter}")

        fig.savefig(os.path.join(out_dir, f"iter_{n_iter:03d}.png"), dpi=120)
        plt_ref.close(fig)

    def _track_evolution_plot(self, snapshots, out_dir):
        import math

        import matplotlib.pyplot as plt

        n = len(snapshots)
        cols = min(n, 3)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for i, (X, labels, centroids, n_iter) in enumerate(snapshots):
            ax = axes[i // cols][i % cols]
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15, alpha=0.6)
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                c=range(self.n_centroids),
                cmap="viridis",
                s=120,
                marker="X",
                edgecolors="black",
                linewidths=1.0,
            )
            ax.set_title(f"Iteración {n_iter}")
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "evolution.png"), dpi=120)
        plt.close(fig)

    def fit(self, X: np.array, track: bool = False, track_dir: str = "kmeans_frames"):
        # X = (N, D)
        self._init_centroids(X.shape[1], np.min(X, axis=0), np.max(X, axis=0))
        iters = 0

        if track:
            import matplotlib

            matplotlib.use("Agg")
            os.makedirs(track_dir, exist_ok=True)
            snapshots = []
            # Si hay más de 2 dims, proyectamos a 2 PCs principales para poder pintarlo
            X_plot = None
            if X.shape[1] > 2:
                X_centered = X - X.mean(axis=0)
                _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
                X_plot = X_centered @ Vt[:2].T

        for iters in range(1, self.max_iters + 1):
            old_centroids = self.centroids.copy()

            # Paso E: Asignación
            dists = self._distance_to_centroids(X)
            labels = self._get_label(dists)

            if track:
                Xsnap = X_plot if X_plot is not None else X
                centroids_snapshot = self.centroids.copy()
                self._track_frame(Xsnap, labels, centroids_snapshot, X_plot, track_dir, iters)
                snapshots.append((Xsnap, labels.copy(), centroids_snapshot, iters))

            # Paso M: Actualización
            self.centroids = self._update_centroids(X, labels)

            # Comprobación de convergencia por tolerancia de desplazamiento
            shift = np.linalg.norm(self.centroids - old_centroids)
            if shift <= self.tol:
                print(f"KMeans convergió tras {iters} iteraciones (shift={shift:.2e}).")
                break

        if track:
            self._track_evolution_plot(snapshots, track_dir)
            print(f"Tracking guardado en '{track_dir}' ({iters} frames + evolution.png).")

        if iters >= self.max_iters:
            print(f"KMeans finalizó tras alcanzar el límite de {self.max_iters} iteraciones.")
        return self



    def predict(self, X: np.array) -> np.array:
        return self._get_label(self._distance_to_centroids(X))

    def fit_predict(self, X: np.array, **kwargs) -> np.array:
        self.fit(X, **kwargs)
        return self.predict(X)

