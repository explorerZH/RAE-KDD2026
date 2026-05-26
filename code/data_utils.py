"""Data loading utilities for RAE.

Supports five datasets used in the paper: CelebA, IMDb, ImageNet (Tiny),
Flickr30k and SIFT1B. The first four ship as pre-computed embedding files;
SIFT1B ships as ``.bvecs`` raw vectors. GIST1M / SIFT1M and other
exploratory datasets present in the original research codebase have been
removed for the public release.
"""
import os
import pickle
import random

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Low-level readers for SIFT1B (.bvecs) and groundtruth (.ivecs) files
# ---------------------------------------------------------------------------

def _read_bvecs(filename, max_count=None, precision='float32'):
    with open(filename, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
    record_bytes = 4 + dim
    total = os.path.getsize(filename) // record_bytes
    if max_count is not None:
        total = min(total, max_count)
    with open(filename, 'rb') as f:
        raw = np.fromfile(f, dtype=np.uint8, count=total * record_bytes)
    raw = raw.reshape(total, record_bytes)
    out_dtype = np.float16 if precision == 'float16' else np.float32
    vectors = raw[:, 4:].astype(out_dtype)
    return vectors


def _read_ivecs(filename, max_count=None):
    with open(filename, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
    record_ints = 1 + dim
    total = os.path.getsize(filename) // (record_ints * 4)
    if max_count is not None:
        total = min(total, max_count)
    with open(filename, 'rb') as f:
        raw = np.fromfile(f, dtype=np.int32, count=total * record_ints)
    return np.ascontiguousarray(raw.reshape(total, record_ints)[:, 1:])


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class VectorDataset(Dataset):
    """Minimal vector container.

    Stores both the raw and L2-normalized copies of the vectors so that
    cosine-distance evaluation does not have to renormalize on every call.
    """

    def __init__(self, vectors, ids, args=None, train_vectors=None,
                 train_ids=None, is_validation=False):
        self.storage_dtype = (
            torch.float16
            if args is not None and getattr(args, 'precision', 'float32') == 'float16'
            else torch.float32
        )
        self.vectors = vectors.to(self.storage_dtype)
        self.normalize_vectors = self._normalize(vectors)
        self.ids = ids
        self.args = args
        self.is_validation = is_validation

        if is_validation:
            assert train_vectors is not None and train_ids is not None
            self.train_vectors = train_vectors.to(self.storage_dtype)
            self.normalize_train_vectors = self._normalize(train_vectors)
            self.train_ids = train_ids
        else:
            self.train_vectors = None
            self.train_ids = None

    def _normalize(self, vectors):
        v = vectors.float()
        n = torch.clamp(torch.norm(v, p=2, dim=1, keepdim=True), min=1e-8)
        return (v / n).to(self.storage_dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return idx


def _collate(batch_indices, dataset):
    use_norm = bool(getattr(dataset.args, 'if_normalize', False))
    src = dataset.normalize_vectors if use_norm else dataset.vectors
    return {'vectors': src[batch_indices].float()}


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_embeddings(data_path, dataset_type, embedding_model_type, num_samples):
    """Returns a dict ``{id: torch.FloatTensor}`` for the four embedding
    datasets (CelebA, IMDb, ImageNet, Flickr30k).
    """
    if dataset_type == 'CelebA':
        fp = os.path.join(data_path, dataset_type, embedding_model_type,
                          'raw_embeddings_withname.pkl')
        with open(fp, 'rb') as f:
            raw = pickle.load(f)
        all_emb = [emb for jpg_dict in raw.values() for emb in jpg_dict.values()]
        num_samples = min(num_samples, len(all_emb))
        idx = random.sample(range(len(all_emb)), num_samples)
        return {i: torch.tensor(all_emb[j], dtype=torch.float32) for i, j in enumerate(idx)}

    if dataset_type == 'imdb':
        fp = os.path.join(data_path, dataset_type, embedding_model_type,
                          'imdb_train_text_embeddings.pkl')
        with open(fp, 'rb') as f:
            raw = pickle.load(f)
        embs = raw['embeddings'][:num_samples]
        return {i: torch.tensor(e, dtype=torch.float32) for i, e in enumerate(embs)}

    if dataset_type == 'flickr30k':
        fp = os.path.join(data_path, dataset_type, embedding_model_type,
                          'flickr30k_embeddings.pkl')
        with open(fp, 'rb') as f:
            raw = pickle.load(f)
        embs = raw['combined_embeddings'][:num_samples]
        return {i: torch.tensor(e, dtype=torch.float32) for i, e in enumerate(embs)}

    if dataset_type == 'ImageNet':
        fp = os.path.join(data_path, dataset_type, embedding_model_type,
                          'ImageNet_embeddings.npz')
        raw = np.load(fp)
        embs = raw['embeddings'][:num_samples]
        return {i: torch.tensor(e, dtype=torch.float32) for i, e in enumerate(embs)}

    raise ValueError(f'Unknown dataset_type: {dataset_type}')


def _load_sift1b(data_path, num_samples, split):
    """Load SIFT1B learn or queries split as a dict."""
    fname = 'learn.bvecs' if split == 'train' else 'queries.bvecs'
    fp = os.path.join(data_path, 'SIFT1B', fname)
    vectors = _read_bvecs(fp, max_count=num_samples)
    return {i: torch.tensor(vectors[i], dtype=torch.float32) for i in range(len(vectors))}


def _load_full_sift1b_base(data_path, precision='float32', max_count=None):
    fp = os.path.join(data_path, 'SIFT1B', 'learn.bvecs')
    return _read_bvecs(fp, max_count=max_count, precision=precision)


def create_data_loaders(data_path, args, val_split=0.1):
    """Build train / val ``DataLoader`` pairs.

    Returns
    -------
    tuple
        For SIFT1B:
            (train_loader, val_loader, train_dataset, val_dataset,
             train_vectors_np, val_vectors_np, full_base_vectors, groundtruth)
        For the embedding datasets the last two entries are ``None``.
    """
    is_sift1b = args.dataset_type == 'SIFT1B'

    if is_sift1b:
        train_dict = _load_sift1b(data_path, args.num_train_samples, 'train')
        val_dict = _load_sift1b(data_path, args.num_val_samples, 'val')
        train_ids = list(train_dict.keys())
        val_ids = list(val_dict.keys())
        train_vectors = torch.stack([train_dict[i] for i in train_ids])
        val_vectors = torch.stack([val_dict[i] for i in val_ids])
        del train_dict, val_dict
    else:
        emb = _load_embeddings(data_path, args.dataset_type,
                               args.embedding_model_type, args.num_samples)
        all_ids = list(emb.keys())
        random.shuffle(all_ids)
        split = int(len(all_ids) * (1 - val_split))
        train_ids, val_ids = all_ids[:split], all_ids[split:]
        train_vectors = torch.stack([emb[i] for i in train_ids])
        val_vectors = torch.stack([emb[i] for i in val_ids])
        del emb

    train_dataset = VectorDataset(train_vectors, train_ids, args=args)
    val_dataset = VectorDataset(val_vectors, val_ids, args=args,
                                train_vectors=train_vectors,
                                train_ids=train_ids,
                                is_validation=True)

    pin = args.device == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=lambda b: _collate(b, train_dataset),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=lambda b: _collate(b, val_dataset),
    )

    print(f'Train set: {len(train_dataset)}, Val set: {len(val_dataset)}')

    full_base_vectors = None
    groundtruth = None
    if is_sift1b:
        full_base_vectors = _load_full_sift1b_base(
            data_path, precision=getattr(args, 'precision', 'float32'),
            max_count=getattr(args, 'num_base_samples', None),
        )
        print(f'SIFT1B base set: {len(full_base_vectors)} vectors '
              f'({full_base_vectors.dtype})')

    return (train_loader, val_loader, train_dataset, val_dataset,
            train_vectors.numpy(), val_vectors.numpy(),
            full_base_vectors, groundtruth)


# ---------------------------------------------------------------------------
# k-NN helpers (used by both train.py and the baselines)
# ---------------------------------------------------------------------------

def compute_knn_order(embeddings, k, distance_metric, ids=None):
    """k-NN within a single set, excluding self."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    n = embeddings.shape[0]
    if ids is None:
        ids = list(range(n))
    if distance_metric == 'euclidean':
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    distances, indices = index.search(embeddings.astype(np.float32), min(k + 1, n))
    out = {}
    for i, id_ in enumerate(ids):
        out[id_] = [(ids[indices[i, j]], float(distances[i, j]))
                    for j in range(1, min(k + 1, n))]
    return out


def compute_knn_cross_set(query_vectors, base_vectors, k, query_ids, base_ids,
                          distance_metric):
    """k-NN of every query in the base set."""
    if isinstance(query_vectors, torch.Tensor):
        query_vectors = query_vectors.cpu().numpy()
    if isinstance(base_vectors, torch.Tensor):
        base_vectors = base_vectors.cpu().numpy()
    if distance_metric == 'euclidean':
        index = faiss.IndexFlatL2(base_vectors.shape[1])
    else:
        index = faiss.IndexFlatIP(base_vectors.shape[1])
    index.add(base_vectors.astype(np.float32))
    distances, indices = index.search(query_vectors.astype(np.float32), k)
    out = {}
    for i, qid in enumerate(query_ids):
        out[qid] = [(base_ids[indices[i, j]], float(distances[i, j]))
                    for j in range(k)]
    return out
