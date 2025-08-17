import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import os
import torch.nn.functional as F

from config import get_args
from model import AutoEncoder
from loss import InfoNCELoss
from data_utils import create_data_loaders, compute_knn_order
import faiss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(base_order, reduction_order, topK_num_list):
    """计算降维前后k近邻的重合准确度"""
    entity_ids = list(base_order.keys())
    
    similarity_dict = {}
    for k in topK_num_list:
        total_similarity = 0
        for id in entity_ids:
            set_a = set(item[0] for item in base_order[id][:k])
            set_b = set(item[0] for item in reduction_order[id][:k])
            
            correct = len(set_b.intersection(set_a))
            similarity = correct / k
            total_similarity += similarity
        
        average_similarity = total_similarity / len(entity_ids)
        similarity_dict[f"top{k}"] = average_similarity
    
    return similarity_dict


def train_epoch(model, train_loader, train_dataset, reconstruction_criterion, 
                contrastive_criterion, optimizer, args, epoch=0):
    """训练一个epoch - 使用预采样的批次"""
    model.train()
    total_loss = 0
    reconstruction_loss_sum = 0
    contrastive_loss_sum = 0
    
    # 渐进式策略：动态更新采样策略
    if args.sample_strategy == 'progressive':
        if epoch < args.warmup_epochs:
            # 预热阶段：随机采样
            strategy = 'random'
            hard_ratio = 0.0
        else:
            # 渐进式增加困难负样本比例
            strategy = 'hard'
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs/3 - args.warmup_epochs)
            hard_ratio = args.progressive_hard_ratio + \
                        (args.hard_negative_ratio - args.progressive_hard_ratio) * progress
            hard_ratio = min(hard_ratio, args.hard_negative_ratio)
        
        # 更新数据集的采样策略（只在策略改变时更新）
        train_dataset.update_sample_strategy(strategy, hard_ratio)
    elif args.sample_strategy == 'hard':
        # 固定困难负样本策略，每个epoch刷新负样本以增加随机性
        if epoch > 0 and epoch % 5 == 0:
            train_dataset.refresh_negative_samples()
    else:
        # 随机策略，每隔几个epoch刷新负样本
        if epoch > 0 and epoch % 10 == 0:
            train_dataset.refresh_negative_samples()
    
    current_strategy = train_dataset.sample_strategy
    current_hard_ratio = train_dataset.hard_negative_ratio if current_strategy == 'hard' else 0.0
    
    progress_bar = tqdm(train_loader, 
                       desc=f"Training (strategy: {current_strategy}, hard_ratio: {current_hard_ratio:.2f})")
    
    for batch in progress_bar:
        # 获取批次数据
        all_vectors = batch['vectors'].to(args.device)
        anchor_indices = batch['anchor_indices'].to(args.device)
        pos_indices = batch['pos_indices'].to(args.device)
        neg_indices = batch['neg_indices'].to(args.device)
        
        optimizer.zero_grad()
        
        # 前向传播：对所有向量进行编码和解码
        all_embeddings, all_reconstructed = model(all_vectors)
        
        # 获取anchor的嵌入和重构
        anchor_embeddings = all_embeddings[anchor_indices]
        anchor_reconstructed = all_reconstructed[anchor_indices]
        anchor_vectors = all_vectors[anchor_indices]
        
        # 计算重构损失（只对anchor计算）
        reconstruction_loss = reconstruction_criterion(anchor_reconstructed, anchor_vectors)
        
        # 计算对比损失（使用向量化版本）
        contrastive_loss = contrastive_criterion(all_embeddings, 
                                                anchor_indices,
                                                pos_indices, 
                                                neg_indices)
        
        # 组合损失
        loss = (1 - args.alpha) * reconstruction_loss + args.alpha * contrastive_loss
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        reconstruction_loss_sum += reconstruction_loss.item()
        contrastive_loss_sum += contrastive_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{reconstruction_loss.item():.4f}',
            'contrast': f'{contrastive_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = reconstruction_loss_sum / len(train_loader)
    avg_contrast_loss = contrastive_loss_sum / len(train_loader)
    
    return avg_loss, avg_recon_loss, avg_contrast_loss


def validate(model, val_loader, reconstruction_criterion, contrastive_criterion, args):
    """验证模型"""
    model.eval()
    total_loss = 0
    reconstruction_loss_sum = 0
    contrastive_loss_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            all_vectors = batch['vectors'].to(args.device)
            anchor_indices = batch['anchor_indices'].to(args.device)
            pos_indices = batch['pos_indices'].to(args.device)
            neg_indices = batch['neg_indices'].to(args.device)
            
            # 前向传播
            all_embeddings, all_reconstructed = model(all_vectors)
            
            # 获取anchor的嵌入和重构
            anchor_embeddings = all_embeddings[anchor_indices]
            anchor_reconstructed = all_reconstructed[anchor_indices]
            anchor_vectors = all_vectors[anchor_indices]
            
            # 计算重构损失
            reconstruction_loss = reconstruction_criterion(anchor_reconstructed, anchor_vectors)
            
            # 计算对比损失
            contrastive_loss = contrastive_criterion(all_embeddings,
                                                    anchor_indices,
                                                    pos_indices,
                                                    neg_indices)
            
            # 组合损失
            loss = (1 - args.alpha) * reconstruction_loss + args.alpha * contrastive_loss
            
            # 记录损失
            total_loss += loss.item()
            reconstruction_loss_sum += reconstruction_loss.item()
            contrastive_loss_sum += contrastive_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon_loss = reconstruction_loss_sum / len(val_loader)
    avg_contrast_loss = contrastive_loss_sum / len(val_loader)
    
    return avg_loss, avg_recon_loss, avg_contrast_loss


def evaluate_knn_preservation(model, train_dataset, val_dataset, args):
    """评估k近邻保持度"""
    model.eval()
    
    results = {}
    
    # 1. 训练集的k近邻保持度（训练集内部）
    train_vectors = train_dataset.vectors.to(args.device)
    with torch.no_grad():
        train_embeddings = model.encode(train_vectors).cpu()
    train_original = train_vectors.cpu()
    
    # 计算训练集内部的k近邻保持度
    max_k = max(args.topk_eval)

    
    train_embeddings = F.normalize(train_embeddings,p=2,dim=1)

    train_original_knn = compute_knn_order(train_original, max_k, train_dataset.ids)
    train_reduced_knn = compute_knn_order(train_embeddings, max_k, train_dataset.ids)
    
    train_accuracy = calculate_accuracy(
        train_original_knn,
        train_reduced_knn,
        args.topk_eval
    )
    
    # 2. 验证集的k近邻保持度（验证集在训练集中的近邻）
    # val_vectors = val_dataset.vectors.to(args.device)
    # with torch.no_grad():
    #     val_embeddings = model.encode(val_vectors).cpu().numpy()
    # val_original = val_vectors.cpu().numpy()
    
    # # 计算验证集在训练集中的k近邻
    # val_original_knn = compute_knn_cross_set(
    #     val_original, train_original, max_k, 
    #     val_dataset.ids, train_dataset.ids
    # )
    # val_reduced_knn = compute_knn_cross_set(
    #     val_embeddings, train_embeddings, max_k,
    #     val_dataset.ids, train_dataset.ids
    # )
    
    # val_accuracy = calculate_accuracy(
    #     val_original_knn,
    #     val_reduced_knn,
    #     args.topk_eval
    # )
    
    results['train'] = train_accuracy
    # results['val'] = val_accuracy
    
    return results


def compute_knn_cross_set(query_vectors, base_vectors, k, query_ids, base_ids):
    """计算查询集在基准集中的k近邻"""
    if isinstance(query_vectors, torch.Tensor):
        query_vectors = query_vectors.cpu().numpy()
    if isinstance(base_vectors, torch.Tensor):
        base_vectors = base_vectors.cpu().numpy()
    
    # 使用FAISS计算k近邻
    index = faiss.IndexFlatL2(base_vectors.shape[1])
    index.add(base_vectors.astype(np.float32))
    
    distances, indices = index.search(query_vectors.astype(np.float32), k)
    
    # 构建k近邻字典
    knn_order = {}
    for i, query_id in enumerate(query_ids):
        neighbors = [(base_ids[indices[i, j]], float(distances[i, j])) 
                    for j in range(k)]
        knn_order[query_id] = neighbors
    
    return knn_order


def main():
    # 解析命令行参数
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化wandb（如果启用）
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, val_loader, train_dataset, val_dataset, vector_dict = create_data_loaders(
        args.data_path, args
    )
    
    # 获取输入维度
    input_dim = train_dataset.vectors.size(1)
    
    # 创建模型
    model = AutoEncoder(
        input_dim=input_dim,
        output_dim=args.output_dim,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        dropout=args.dropout
    ).to(args.device)
    
    # 创建损失函数
    reconstruction_criterion = nn.MSELoss()
    
    # 使用InfoNCE损失（向量化版本）
    contrastive_criterion = InfoNCELoss(temperature=args.temperature)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 训练循环
    best_val_loss = float('inf')
    best_accuracy = defaultdict(float)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss, train_recon_loss, train_contrast_loss = train_epoch(
            model, train_loader, train_dataset, reconstruction_criterion,
            contrastive_criterion, optimizer, args, epoch
        )
        
        # 验证
        val_loss, val_recon_loss, val_contrast_loss = validate(
            model, val_loader, reconstruction_criterion,
            contrastive_criterion, args
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, "
              f"Contrast: {train_contrast_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, "
              f"Contrast: {val_contrast_loss:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 定期评估k近邻保持度
        if (epoch + 1) % args.eval_interval == 0:
            print("\n评估k近邻保持度...")
            accuracy_results = evaluate_knn_preservation(model, train_dataset, val_dataset, args)
            
            print("训练集k近邻保持准确度:")
            for k, acc in accuracy_results['train'].items():
                print(f"  {k}: {acc:.4f}")
            
            # print("验证集k近邻保持准确度:")
            # for k, acc in accuracy_results['val'].items():
            #     print(f"  {k}: {acc:.4f}")
            
            # 记录到wandb
            if args.use_wandb:
                wandb.log({
                    f'train_accuracy/{k}': acc
                    for k, acc in accuracy_results['train'].items()
                })
                # wandb.log({
                #     f'val_accuracy/{k}': acc
                #     for k, acc in accuracy_results['val'].items()
                # })
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/reconstruction_loss': train_recon_loss,
                'train/contrastive_loss': train_contrast_loss,
                'val/loss': val_loss,
                'val/reconstruction_loss': val_recon_loss,
                'val/contrastive_loss': val_contrast_loss,
                'lr': scheduler.get_last_lr()[0],
                'epoch': epoch + 1
            })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print("保存最佳模型!")
    
    # 最终评估
    print("\n最终评估...")
    final_accuracy = evaluate_knn_preservation(model, train_dataset, val_dataset, args)
    
    print("\n最终训练集k近邻保持准确度:")
    for k, acc in final_accuracy['train'].items():
        print(f"  {k}: {acc:.4f}")
    
    print("\n最终验证集k近邻保持准确度:")
    for k, acc in final_accuracy['val'].items():
        print(f"  {k}: {acc:.4f}")
    
    # 保存最终结果
    results = {
        'args': vars(args),
        'best_val_loss': best_val_loss,
        'final_train_accuracy': final_accuracy['train'],
        'final_val_accuracy': final_accuracy['val']
    }
    
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    if args.use_wandb:
        wandb.finish()
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()