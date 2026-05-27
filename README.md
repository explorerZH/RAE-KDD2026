# RAE: Regularized Auto-Encoder for k-NN Preserving Dimensionality Reduction

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.32413305-blue)](https://doi.org/10.6084/m9.figshare.32413305)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for the paper

> Han Zhang and Dongfang Zhao.
> *RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search.*
> KDD 2026.

RAE learns a linear encoder / decoder pair with a Frobenius-norm penalty on
the encoder matrix. The penalty controls the condition number of the
encoder, which the paper shows is the quantity that governs k-nearest
neighbor preservation under dimensionality reduction (Section 3.4,
Eqs. 19–20). Empirically RAE outperforms PCA, UMAP, Isomap, MDS, RP and
LPP on five datasets (Tables 1–2 in the paper).

## Repository layout

```
code/
  config.py        # argument parser (shared between RAE and baselines)
  model.py         # linear encoder / decoder
  data_utils.py    # dataset loaders + FAISS-based k-NN helpers
  train.py         # train + evaluate RAE
  baselines.py     # PCA, UMAP, Isomap, MDS, RP, LPP baselines
requirements.txt
LICENSE
```

## Installation

```bash
conda create -n rae python=3.11
conda activate rae
pip install -r requirements.txt
```

`faiss-cpu` is sufficient for all experiments in the paper. A GPU is
recommended but not required; the RAE encoder is a single linear layer
so training takes only a few seconds even on CPU.

## Data

The five datasets used in the paper are **not redistributed here** because
they originate from third-party sources with their own licenses. Please
download them yourself:

| Dataset       | Source                                                        | Embedding model used in paper |
|---------------|---------------------------------------------------------------|-------------------------------|
| CelebA        | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html             | ViT (512d)                    |
| IMDb          | https://ai.stanford.edu/~amaas/data/sentiment/                | MPNet (768d)                  |
| ImageNet-Tiny | https://www.kaggle.com/c/tiny-imagenet                        | DINOv2 (384d)                 |
| Flickr30k     | https://huggingface.co/datasets/nlphuji/flickr30k             | CLIP, img+txt concat (1024d)  |
| SIFT1B        | http://corpus-texmex.irisa.fr/                                | raw (128d)                    |

The loaders in `code/data_utils.py` expect the following layout under
`--data_path`:

```
data/
  CelebA/CLIP(VIT)/raw_embeddings_withname.pkl
  imdb/mpnet/imdb_train_text_embeddings.pkl
  ImageNet/DINOv2/ImageNet_embeddings.npz
  flickr30k/CLIP(VIT)/flickr30k_embeddings.pkl
  SIFT1B/learn.bvecs
  SIFT1B/queries.bvecs
```

Expected internal format of each embedding file:

| File | Structure |
|------|-----------|
| `CelebA/CLIP(VIT)/raw_embeddings_withname.pkl` | `dict[identity_id -> dict[jpg_name -> np.ndarray(512,)]]` |
| `imdb/mpnet/imdb_train_text_embeddings.pkl`    | `dict` with key `'embeddings'`: `list/array` of shape `(N, 768)` |
| `ImageNet/DINOv2/ImageNet_embeddings.npz`      | NPZ with key `'embeddings'`: `np.ndarray` of shape `(N, 384)` |
| `flickr30k/CLIP(VIT)/flickr30k_embeddings.pkl` | `dict` with key `'combined_embeddings'`: `list/array` of `(N, 1024)` (image + text features concatenated) |
| `SIFT1B/learn.bvecs`, `SIFT1B/queries.bvecs`   | Standard TEXMEX `.bvecs` (uint8) — used as-is |

Producing the four embedding files is a one-off step: run the corresponding
pre-trained encoder (CLIP-ViT for CelebA / Flickr30k, MPNet for IMDb,
DINOv2 for ImageNet) on the raw inputs and dump in the schema above.

## Reproducing the main results

Train RAE (Table 1):

```bash
python code/train.py --dataset_type CelebA --embedding_model_type "CLIP(VIT)" \
    --num_samples 10000 --output_dim 256 --distance_metric cosine \
    --weight_decay 2e-5 --steps 3000
```

`--weight_decay` corresponds to the regularization coefficient λ in
Eq. (7).

Run a baseline (Table 1):

```bash
python code/baselines.py --method PCA --dataset_type CelebA \
    --embedding_model_type "CLIP(VIT)" --num_samples 10000 \
    --output_dim 256 --distance_metric cosine
```

`--method` accepts `PCA`, `UMAP`, `ISOMAP`, `MDS`, `RP`, `LPP`.

For SIFT1B (Table 2), use `--dataset_type SIFT1B` with `--num_train_samples`,
`--num_val_samples` and (optionally) `--num_base_samples`.

## Output

Both `train.py` and `baselines.py` write a JSON file under `./checkpoints`
or `./results` containing the full set of `topk` accuracies (Eq. 4 in the
paper) and the wall-clock timings (Section 4.4).

## Citation

```bibtex
@inproceedings{zhang2026rae,
  title     = {RAE: A Neural Network Dimensionality Reduction Method for
               Nearest Neighbors Preservation in Vector Search},
  author    = {Zhang, Han and Zhao, Dongfang},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining (KDD '26)},
  year      = {2026},
  doi       = {10.1145/nnnnnnn.nnnnnnn}
}
```

## License

MIT — see [LICENSE](LICENSE).
