# RAE: Regularized Auto-Encoder for k-NN Preserving Dimensionality Reduction

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

For the embedding datasets, each file is a pickled dict / NPZ produced
by the corresponding pre-trained model (CLIP, MPNet, DINOv2). For SIFT1B
the raw `.bvecs` files from the TEXMEX distribution are used directly.

## Reproducing the main results

### RAE (Table 1)

```bash
# Example: CelebA, reduce 512 -> 256, cosine evaluation
python code/train.py \
    --dataset_type CelebA --embedding_model_type "CLIP(VIT)" \
    --num_samples 10000 \
    --output_dim 256 --distance_metric cosine \
    --weight_decay 1e-7 --optimizer Adam --lr 1e-3 --steps 3000
```

The `--weight_decay` argument is the regularization coefficient
λ in Eq. (7). The optimal range varies by dataset; the values used in
the paper are reported in Section 4.3 and can be reproduced by sweeping
λ ∈ {0, 1e-9, …, 1e-6}.

### Baselines (Table 1)

```bash
python code/baselines.py --method PCA --dataset_type CelebA \
    --embedding_model_type "CLIP(VIT)" --num_samples 10000 \
    --output_dim 256 --distance_metric cosine
```

Replace `--method` with one of: `PCA`, `UMAP`, `ISOMAP`, `MDS`, `RP`, `LPP`.

### SIFT1B (Table 2)

```bash
# RAE
python code/train.py --dataset_type SIFT1B --output_dim 64 \
    --num_train_samples 30000 --num_val_samples 10000 \
    --distance_metric cosine --weight_decay 1e-7 --steps 3000

# PCA baseline at the same scale
python code/baselines.py --method PCA --dataset_type SIFT1B \
    --output_dim 64 --distance_metric cosine \
    --num_train_samples 30000 --num_val_samples 10000
```

To match the paper's 50M base-set setting, set `--num_base_samples 50000000`.

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
