"""Configuration / argument parsing for RAE training.

Only the arguments actually used by the released RAE pipeline are kept:
contrastive-learning, hard-negative sampling and InfoNCE knobs from the
exploratory codebase have been removed.
"""
import argparse
import ast
import torch


def str_to_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(s)


def get_args():
    parser = argparse.ArgumentParser(
        description='RAE: Regularized Auto-Encoder for k-NN preserving dimensionality reduction'
    )

    # Data
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Root directory containing dataset folders')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Target (reduced) dimensionality m')
    parser.add_argument('--dataset_type', type=str, default='CelebA',
                        choices=['CelebA', 'imdb', 'ImageNet', 'flickr30k', 'SIFT1B'],
                        help='Dataset identifier')
    parser.add_argument('--embedding_model_type', type=str, default='CLIP(VIT)',
                        help='Subfolder name identifying the embedding model'
                             ' (ignored for SIFT1B, which ships raw vectors)')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of vectors used (for non-SIFT1B datasets)')
    parser.add_argument('--num_train_samples', type=int, default=30000,
                        help='Training subset size (SIFT1B)')
    parser.add_argument('--num_val_samples', type=int, default=10000,
                        help='Validation subset size (SIFT1B)')
    parser.add_argument('--num_base_samples', type=int, default=None,
                        help='Optional cap on the base set size for SIFT1B evaluation.'
                             ' Default None loads the full base set.')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Distance metric used for k-NN evaluation')

    # Model (single linear layer encoder/decoder by default; the paper uses
    # hidden_dims=[] = pure linear RAE).
    parser.add_argument('--hidden_dims', type=str, default='[]',
                        help='Optional hidden layer widths, e.g. "[512]". Default [] = linear.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['relu', 'tanh', 'sigmoid'])

    # Optimisation
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=3000,
                        help='Total number of gradient update steps')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Frobenius-norm regularization coefficient lambda'
                             ' (implemented via optimizer weight_decay)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'AdamW', 'SGD'])

    # Evaluation
    parser.add_argument('--topk_eval', type=str, default='[1,5,10,20,50]')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Run k-NN preservation evaluation every N epochs')

    # Baseline selector (only used by baselines.py)
    parser.add_argument('--method', type=str, default='PCA',
                        choices=['PCA', 'UMAP', 'ISOMAP', 'MDS', 'RP', 'LPP'],
                        help='Baseline method to run (baselines.py only)')

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_last_model', type=lambda x: str(x).lower() == 'true',
                        default=False)
    parser.add_argument('--if_normalize', type=lambda x: str(x).lower() == 'true',
                        default=False,
                        help='Train on L2-normalized vectors (recommended for cosine eval)')
    parser.add_argument('--precision', type=str, default='float32',
                        choices=['float16', 'float32'],
                        help='Vector storage precision; float16 halves memory for SIFT1B')

    # W&B (optional; off by default)
    parser.add_argument('--use_wandb', type=lambda x: str(x).lower() == 'true',
                        default=False)
    parser.add_argument('--wandb_project', type=str, default='rae')
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()
    args.hidden_dims = str_to_list(args.hidden_dims)
    args.topk_eval = str_to_list(args.topk_eval)
    return args
