"""Train the RAE model.

Implements the loss in Section 3.2 of the paper:

    L = (1/N) * sum_i || W_d W_e x_i - x_i ||_2^2 + lambda * || W_e ||_F^2

The Frobenius-norm regularization on the encoder is implemented via the
optimizer's ``weight_decay``. Adam / AdamW with ``weight_decay=lambda``
yields a gradient update term identical to differentiating the explicit
penalty (see Eq. (10) in the paper).
"""
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import get_args
from data_utils import (compute_knn_cross_set, compute_knn_order,
                        create_data_loaders)
from model import AutoEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_accuracy(base_order, reduced_order, topk_list):
    """Compute P_overall (Eq. (4)) for each top-k."""
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    pb = tqdm(loader, desc='Training', leave=False)
    for batch in pb:
        x = batch['vectors'].to(device)
        optimizer.zero_grad()
        _, x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total += loss.item()
        pb.set_postfix(loss=f'{loss.item():.4e}')
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc='Validation', leave=False):
        x = batch['vectors'].to(device)
        _, x_hat = model(x)
        total += criterion(x_hat, x).item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_knn(model, train_dataset, val_dataset, args,
                 full_base_vectors=None, skip_train_eval=False):
    """k-NN preservation evaluation on train / val splits."""
    model.eval()
    results = {}

    def _vec(ds):
        return (ds.normalize_vectors if args.distance_metric == 'cosine'
                else ds.vectors).float().to(args.device)

    max_k = max(args.topk_eval)

    if not skip_train_eval:
        tv = _vec(train_dataset)
        te = model.encode(tv).cpu()
        if args.distance_metric == 'cosine':
            te = F.normalize(te, p=2, dim=1)
        tv_cpu = tv.cpu()
        orig = compute_knn_order(tv_cpu, max_k, args.distance_metric, train_dataset.ids)
        red = compute_knn_order(te, max_k, args.distance_metric, train_dataset.ids)
        results['train'] = topk_accuracy(orig, red, args.topk_eval)
    else:
        results['train'] = None

    vv = _vec(val_dataset)
    ve = model.encode(vv).cpu()
    if args.distance_metric == 'cosine':
        ve = F.normalize(ve, p=2, dim=1)
    vv_cpu = vv.cpu().numpy()

    if full_base_vectors is not None:
        # SIFT1B: project the full base set in chunks
        n = len(full_base_vectors)
        base_tensor = torch.from_numpy(full_base_vectors)
        if args.distance_metric == 'cosine':
            out_dtype = torch.float16 if full_base_vectors.dtype == np.float16 else torch.float32
            normed = torch.empty_like(base_tensor, dtype=out_dtype)
            for i in range(0, n, 100_000):
                j = min(i + 100_000, n)
                b = base_tensor[i:j].float()
                normed[i:j] = (b / torch.clamp(torch.norm(b, p=2, dim=1, keepdim=True), min=1e-8)).to(out_dtype)
            base_tensor = normed
        base_emb = torch.empty((n, args.output_dim), dtype=torch.float32)
        for i in range(0, n, 100_000):
            j = min(i + 100_000, n)
            b = base_tensor[i:j].float().to(args.device)
            base_emb[i:j] = model.encode(b).cpu()
        del base_tensor
        if args.distance_metric == 'cosine':
            base_emb = F.normalize(base_emb, p=2, dim=1)
        base_emb_np = base_emb.numpy()
        base_ids = list(range(n))
        base_f32 = full_base_vectors.astype(np.float32) if full_base_vectors.dtype != np.float32 else full_base_vectors
        orig = compute_knn_cross_set(vv_cpu, base_f32, max_k,
                                     val_dataset.ids, base_ids, args.distance_metric)
        red = compute_knn_cross_set(ve, base_emb_np, max_k,
                                    val_dataset.ids, base_ids, args.distance_metric)
    else:
        if skip_train_eval:
            tv = _vec(train_dataset)
            te = model.encode(tv).cpu()
            if args.distance_metric == 'cosine':
                te = F.normalize(te, p=2, dim=1)
            tv_cpu = tv.cpu()
        orig = compute_knn_cross_set(vv_cpu, tv_cpu, max_k,
                                     val_dataset.ids, train_dataset.ids,
                                     args.distance_metric)
        red = compute_knn_cross_set(ve, te, max_k,
                                    val_dataset.ids, train_dataset.ids,
                                    args.distance_metric)

    results['val'] = topk_accuracy(orig, red, args.topk_eval)
    return results


def build_optimizer(args, model):
    if args.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def main():
    args = get_args()
    set_seed(args.seed)

    is_sift1b = args.dataset_type == 'SIFT1B'

    save_dir = os.path.join(
        args.save_dir,
        f'{args.dataset_type}_{args.embedding_model_type}_{args.num_samples}',
    )
    os.makedirs(save_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f'RAE_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                   config=vars(args))

    result = create_data_loaders(args.data_path, args)
    (train_loader, val_loader, train_dataset, val_dataset,
     _, _, full_base_vectors, _) = result

    input_dim = train_dataset.vectors.size(1)
    model = AutoEncoder(
        input_dim=input_dim, output_dim=args.output_dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation, dropout=args.dropout,
    ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(args, model)
    epochs = max(1, args.steps // max(1, len(train_loader)))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.lr * 0.01,
    )

    print(f'Input dim {input_dim} -> output dim {args.output_dim} | '
          f'lambda (weight_decay) = {args.weight_decay} | '
          f'optimizer = {args.optimizer} | epochs = {epochs}')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        scheduler.step()
        print(f'Epoch {epoch + 1}/{epochs} | train {train_loss:.4e} | val {val_loss:.4e} | '
              f'lr {scheduler.get_last_lr()[0]:.2e}')

        if args.use_wandb:
            import wandb
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
                       'lr': scheduler.get_last_lr()[0]}, step=epoch + 1)

        if (epoch + 1) % args.eval_interval == 0:
            acc = evaluate_knn(model, train_dataset, val_dataset, args,
                               full_base_vectors=full_base_vectors,
                               skip_train_eval=is_sift1b)
            if acc['train'] is not None:
                print('  train accuracy:', acc['train'])
            print('  val   accuracy:', acc['val'])
            if args.use_wandb:
                import wandb
                if acc['train'] is not None:
                    wandb.log({f'train_acc/{k}': v for k, v in acc['train'].items()},
                              step=epoch + 1)
                wandb.log({f'val_acc/{k}': v for k, v in acc['val'].items()},
                          step=epoch + 1)

    print('\nFinal evaluation...')
    final = evaluate_knn(model, train_dataset, val_dataset, args,
                         full_base_vectors=full_base_vectors,
                         skip_train_eval=is_sift1b)
    if final['train'] is not None:
        print('Final train accuracy:', final['train'])
    print('Final val   accuracy:', final['val'])

    out = {'args': vars(args),
           'final_train_accuracy': final['train'],
           'final_val_accuracy': final['val']}
    out_name = (f'{args.output_dim}d_{args.distance_metric}_{args.optimizer}_'
                f'{args.weight_decay}_normalize{args.if_normalize}_results.json')
    with open(os.path.join(save_dir, out_name), 'w') as f:
        json.dump(out, f, indent=2)

    if args.save_last_model:
        ckpt_name = (f'{args.output_dim}d_{args.optimizer}_{args.weight_decay}'
                     f'_normalize{args.if_normalize}_model.ckpt')
        torch.save({'model_state_dict': model.state_dict(),
                    'args': vars(args)},
                   os.path.join(save_dir, ckpt_name))

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
