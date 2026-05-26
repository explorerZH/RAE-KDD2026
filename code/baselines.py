"""Baseline dimensionality reduction methods reported in the paper.

Supports the six baselines compared against RAE:
    PCA, UMAP, ISOMAP, MDS (with Ridge out-of-sample extension),
    RP (Random Projection), LPP (Locality Preserving Projections).

Usage
-----
    python baselines.py --method PCA --dataset_type CelebA \
        --embedding_model_type "CLIP(VIT)" --output_dim 256 --distance_metric cosine

For SIFT1B, only PCA and RP are computationally feasible at scale.
"""
import json
import os
import random
import time
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from config import get_args
from data_utils import (compute_knn_cross_set, compute_knn_order,
                        create_data_loaders)


# ---------------------------------------------------------------------------
# Reducer implementations
# ---------------------------------------------------------------------------

class Reducer:
    """Common interface implemented by every baseline."""
    name = 'base'

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class PCAReducer(Reducer):
    name = 'PCA'

    def __init__(self, n_components, seed):
        from sklearn.decomposition import PCA
        self.model = PCA(n_components=n_components, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)


class UMAPReducer(Reducer):
    name = 'UMAP'

    def __init__(self, n_components, seed):
        from umap import UMAP
        self.model = UMAP(n_components=n_components, n_neighbors=5,
                          min_dist=0.2, metric='euclidean',
                          random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)


class IsomapReducer(Reducer):
    name = 'ISOMAP'

    def __init__(self, n_components, seed):
        from sklearn.manifold import Isomap
        self.model = Isomap(n_components=n_components, n_neighbors=10, n_jobs=-1)

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)


class MDSReducer(Reducer):
    """Metric MDS with Ridge regression as the out-of-sample extension."""
    name = 'MDS'

    def __init__(self, n_components, seed, max_samples=5000):
        from sklearn.manifold import MDS
        from sklearn.linear_model import Ridge
        self.model = MDS(n_components=n_components, n_init=1, max_iter=400,
                         random_state=seed, dissimilarity='precomputed',
                         n_jobs=-1)
        self.reg = Ridge(alpha=1.0)
        self.max_samples = max_samples
        self.seed = seed

    def fit(self, X):
        from sklearn.metrics import pairwise_distances
        if X.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(X.shape[0], self.max_samples, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        Z = self.model.fit_transform(pairwise_distances(X_sub, metric='euclidean'))
        self.reg.fit(X_sub, Z)
        return self

    def transform(self, X):
        return self.reg.predict(X)


class RPReducer(Reducer):
    """Gaussian random projection, sqrt(d_in / d_out) scaling per JL."""
    name = 'RP'

    def __init__(self, n_components, seed):
        self.n_components = n_components
        self.seed = seed
        self.R = None

    def fit(self, X):
        d_in = X.shape[1]
        rng = np.random.default_rng(self.seed)
        self.R = rng.standard_normal((d_in, self.n_components)).astype(np.float32)
        self.R /= np.sqrt(self.n_components)
        return self

    def transform(self, X):
        return X.astype(np.float32) @ self.R


class LPPReducer(Reducer):
    """Linear LPP via the generalized eigenproblem X^T L X a = lambda X^T D X a."""
    name = 'LPP'

    def __init__(self, n_components, seed, k_neighbors=10):
        self.n_components = n_components
        self.k = k_neighbors
        self.seed = seed
        self.A = None

    def fit(self, X):
        from scipy import sparse
        from scipy.linalg import eigh
        X = X.astype(np.float32)
        n, d = X.shape
        # Build k-NN graph with FAISS
        index = faiss.IndexFlatL2(d)
        index.add(X)
        _, idx = index.search(X, self.k + 1)
        rows = np.repeat(np.arange(n), self.k)
        cols = idx[:, 1:].reshape(-1)
        data = np.ones_like(rows, dtype=np.float32)
        W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
        W = (W + W.T).minimum(1.0).tocsr()  # symmetrize
        D = np.asarray(W.sum(axis=1)).ravel()
        L = sparse.diags(D) - W
        # Solve X^T L X a = lambda X^T D X a as a dense generalized eigenproblem.
        # Cast to float64 up front so the ridge isn't dwarfed by accumulated noise.
        A = (X.T @ (L @ X)).astype(np.float64)
        B = (X.T @ (sparse.diags(D) @ X)).astype(np.float64)
        # Tikhonov ridge scaled to the trace of B for positive-definiteness;
        # required when d_in is large compared with n (e.g. CLIP 512d on 2k samples).
        ridge = 1e-4 * (np.trace(B) / d)
        B += ridge * np.eye(d, dtype=np.float64)
        eigvals, eigvecs = eigh(A, B)
        # Smallest non-trivial eigenvalues
        self.A = eigvecs[:, :self.n_components].astype(np.float32)
        return self

    def transform(self, X):
        return X.astype(np.float32) @ self.A


REDUCERS = {
    'PCA': PCAReducer, 'UMAP': UMAPReducer, 'ISOMAP': IsomapReducer,
    'MDS': MDSReducer, 'RP': RPReducer, 'LPP': LPPReducer,
}


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + 1e-8)


def _topk_accuracy(base_order, reduced_order, topk_list):
    keys = list(base_order.keys())
    out = {}
    for k in topk_list:
        s = 0.0
        for key in keys:
            a = {item[0] for item in base_order[key][:k]}
            b = {item[0] for item in reduced_order[key][:k]}
            s += len(a & b) / k
        out[f'top{k}'] = s / len(keys)
    return out


def evaluate(reducer, train_vectors, val_vectors, train_ids, val_ids, args,
             full_base_vectors=None):
    if args.distance_metric == 'cosine':
        train_in = _normalize(train_vectors)
        val_in = _normalize(val_vectors)
    else:
        train_in = train_vectors.astype(np.float32)
        val_in = val_vectors.astype(np.float32)

    t0 = time.time()
    reducer.fit(train_in)
    train_time = time.time() - t0

    t0 = time.time()
    train_red = reducer.transform(train_in)
    val_red = reducer.transform(val_in)
    infer_time = time.time() - t0

    if args.distance_metric == 'cosine':
        train_red = _normalize(train_red)
        val_red = _normalize(val_red)

    max_k = max(args.topk_eval)
    results = {'train_time': train_time, 'inference_time': infer_time}

    if full_base_vectors is not None:
        # SIFT1B / large-scale path
        base_in = (_normalize(full_base_vectors.astype(np.float32))
                   if args.distance_metric == 'cosine'
                   else full_base_vectors.astype(np.float32))
        base_red = reducer.transform(base_in)
        if args.distance_metric == 'cosine':
            base_red = _normalize(base_red)
        base_ids = list(range(len(base_in)))
        orig = compute_knn_cross_set(val_in, base_in, max_k, val_ids, base_ids,
                                     args.distance_metric)
        red = compute_knn_cross_set(val_red, base_red, max_k, val_ids, base_ids,
                                    args.distance_metric)
        results['val'] = _topk_accuracy(orig, red, args.topk_eval)
    else:
        # In-sample (train internal)
        orig_t = compute_knn_order(train_in, max_k, args.distance_metric, train_ids)
        red_t = compute_knn_order(train_red, max_k, args.distance_metric, train_ids)
        results['train'] = _topk_accuracy(orig_t, red_t, args.topk_eval)
        # Cross-set (val vs. train)
        orig_v = compute_knn_cross_set(val_in, train_in, max_k, val_ids,
                                       train_ids, args.distance_metric)
        red_v = compute_knn_cross_set(val_red, train_red, max_k, val_ids,
                                      train_ids, args.distance_metric)
        results['val'] = _topk_accuracy(orig_v, red_v, args.topk_eval)

    return results


def main():
    args = get_args()
    method = args.method

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Running baseline: {method}')
    result = create_data_loaders(args.data_path, args)
    (_, _, train_dataset, val_dataset, train_np, val_np,
     full_base_vectors, _) = result

    reducer = REDUCERS[method](args.output_dim, args.seed)
    metrics = evaluate(reducer, train_np, val_np,
                       train_dataset.ids, val_dataset.ids, args,
                       full_base_vectors=full_base_vectors)

    print(json.dumps(metrics, indent=2))

    os.makedirs('./results', exist_ok=True)
    out_path = (f'./results/{method}_{args.dataset_type}_{args.embedding_model_type}_'
                f'{args.output_dim}d_{args.distance_metric}_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'method': method, **metrics}, f, indent=2)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
