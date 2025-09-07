import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import faiss
from tqdm import tqdm
import os
import pickle
import random

class VectorDataset(Dataset):
    """
    向量数据集类 - 改进版本，支持批次时预采样
    """
    def __init__(self, vectors, ids, args = None, k_neighbors=10, neg_sample_ratio=5,
                 hard_negative_ratio=0.5, hard_negative_range=10,
                 sample_strategy='random', initial_hard_ratio=None,
                 train_vectors=None, train_ids=None, is_validation=False):
        """
        Args:
            vectors: 向量矩阵 [n_samples, dim]
            ids: ID列表
            k_neighbors: k近邻数量
            neg_sample_ratio: 负样本比例
            hard_negative_ratio: 困难负样本比例（目标比例）
            hard_negative_range: 困难负样本范围
            sample_strategy: 采样策略
            initial_hard_ratio: 初始困难负样本比例（用于渐进式策略）
            train_vectors: 训练集向量（仅验证集使用）
            train_ids: 训练集ID（仅验证集使用）
            is_validation: 是否为验证集
        """
        # 归一化向量
        self.vectors = vectors
        self.normalize_vectors = self._normalize_vectors(vectors)
        self.ids = ids
        self.args = args
        self.k_neighbors = k_neighbors
        self.neg_sample_ratio = neg_sample_ratio
        self.target_hard_negative_ratio = hard_negative_ratio
        self.hard_negative_ratio = initial_hard_ratio if initial_hard_ratio is not None else hard_negative_ratio
        self.hard_negative_range = hard_negative_range
        self.sample_strategy = sample_strategy
        self.is_validation = is_validation
        
        # 如果是验证集，保存训练集信息
        if is_validation:
            assert train_vectors is not None and train_ids is not None, \
                "验证集需要提供训练集向量和ID"
            self.train_vectors = train_vectors
            self.normalize_train_vectors = self._normalize_vectors(train_vectors)
            self.train_ids = train_ids
        else:
            self.train_vectors = None
            self.train_ids = None
        
        # 计算k近邻和相似度
        if self.sample_strategy != 'pure_ae':
            print(f"计算k近邻标签矩阵（{'验证集->训练集' if is_validation else '训练集内部'}）...")
            self.knn_indices, self.knn_similarities = self._compute_knn_indices_and_similarities()
            
            # 预计算每个样本的正负样本索引
            print(f"预计算正负样本索引...")
            self.pos_indices, self.neg_indices = self._precompute_pos_neg_indices()
        else:
            print("当前为纯auto-encoder学习，无需额外采样策略")
    
    def _normalize_vectors(self, vectors):
        """对向量进行L2归一化"""
        norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        return vectors / norms
    
    def _compute_knn_indices_and_similarities(self):
        """计算k近邻索引和相似度"""
        if self.is_validation:
            # 验证集：在训练集中寻找k近邻
            query_vectors = self.vectors.cpu().numpy()
            base_vectors = self.train_vectors.cpu().numpy()
            n_query = len(query_vectors)
            n_base = len(base_vectors)
            d = query_vectors.shape[1]
            
            # 使用内积搜索（归一化向量的余弦相似度）
            if self.args.distance_metric == 'euclidean':
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(base_vectors.astype(np.float32))
            
            # 搜索k+h近邻
            k_search = min(self.k_neighbors + self.hard_negative_range, n_base)
            similarities, indices = index.search(query_vectors.astype(np.float32), k_search)
            
            return torch.tensor(indices, dtype=torch.long), torch.tensor(similarities)
        else:
            # 训练集：在自身中寻找k近邻
            vectors_np = self.vectors.cpu().numpy()
            n_samples = len(vectors_np)
            d = vectors_np.shape[1]
            
            # 使用内积搜索
            if self.args.distance_metric == 'euclidean':
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(vectors_np.astype(np.float32))
            
            # 搜索k+h+1近邻（包括自身）
            k_search = min(self.k_neighbors + self.hard_negative_range + 1, n_samples)
            similarities, indices = index.search(vectors_np.astype(np.float32), k_search)
            
            # 移除自身（第0个），保留k近邻和潜在的困难负样本
            knn_indices = indices[:, 1:]
            knn_similarities = similarities[:, 1:]
            
            return torch.tensor(knn_indices, dtype=torch.long), torch.tensor(knn_similarities)
    
    def _precompute_pos_neg_indices(self):
        """预计算每个样本的正负样本索引"""
        n_samples = len(self.vectors)
        num_neg = int(self.k_neighbors * self.neg_sample_ratio)
        
        pos_indices = []
        neg_indices = []
        
        for i in range(n_samples):
            # 正样本：前k个近邻
            pos_idx = self.knn_indices[i, :self.k_neighbors]
            pos_indices.append(pos_idx)
            
            # 负样本采样
            if self.sample_strategy == 'hard':
                neg_idx = self._sample_hard_negatives(i, num_neg)
            else:
                neg_idx = self._sample_random_negatives(i, num_neg)
            neg_indices.append(neg_idx)
        
        return pos_indices, neg_indices
    
    def _sample_random_negatives(self, anchor_idx, num_neg):
        """随机采样负样本"""
        if self.is_validation:
            # 验证集：从训练集中采样负样本
            n_samples = len(self.train_vectors)
            # 获取正样本集合（在训练集中的索引）
            pos_set = set(self.knn_indices[anchor_idx, :self.k_neighbors].tolist())
        else:
            # 训练集：从自身中采样负样本
            n_samples = len(self.vectors)
            # 获取正样本集合
            pos_set = set(self.knn_indices[anchor_idx, :self.k_neighbors].tolist())
            pos_set.add(anchor_idx)  # 排除自身
        
        # 所有可能的负样本
        neg_candidates = [i for i in range(n_samples) if i not in pos_set]
        
        # 随机采样
        if len(neg_candidates) > num_neg:
            neg_idx = random.sample(neg_candidates, num_neg)
        else:
            neg_idx = neg_candidates
        
        return torch.tensor(neg_idx, dtype=torch.long)
    
    def _sample_hard_negatives(self, anchor_idx, num_neg):
        """困难负样本挖掘"""
        if self.is_validation:
            # 验证集：从训练集中采样困难负样本
            n_samples = len(self.train_vectors)
            # 获取正样本集合（在训练集中的索引）
            pos_set = set(self.knn_indices[anchor_idx, :self.k_neighbors].tolist())
        else:
            # 训练集：从自身中采样困难负样本
            n_samples = len(self.vectors)
            # 获取正样本集合
            pos_set = set(self.knn_indices[anchor_idx, :self.k_neighbors].tolist())
            pos_set.add(anchor_idx)  # 排除自身
        
        num_hard = int(num_neg * self.hard_negative_ratio)
        num_random = num_neg - num_hard
        
        # 困难负样本：k到k+h范围内的近邻
        hard_candidates = []
        for j in range(self.k_neighbors, min(self.k_neighbors + self.hard_negative_range, 
                                            self.knn_indices.shape[1])):
            idx = self.knn_indices[anchor_idx, j].item()
            if idx not in pos_set:
                hard_candidates.append(idx)
        
        # 采样困难负样本
        if len(hard_candidates) > num_hard:
            hard_neg = random.sample(hard_candidates, num_hard)
        else:
            hard_neg = hard_candidates
            num_random = num_neg - len(hard_neg)
        
        # 随机负样本
        all_neg_candidates = [i for i in range(n_samples) if i not in pos_set]
        remaining_candidates = [i for i in all_neg_candidates if i not in hard_neg]
        
        if len(remaining_candidates) > num_random and num_random > 0:
            random_neg = random.sample(remaining_candidates, num_random)
        else:
            random_neg = remaining_candidates[:num_random] if num_random > 0 else []
        
        neg_idx = hard_neg + random_neg
        return torch.tensor(neg_idx, dtype=torch.long)
    
    def update_sample_strategy(self, strategy, hard_ratio=None):
        """动态更新采样策略（用于渐进式训练）"""
        # 验证集不需要更新策略
        if self.is_validation:
            return
        
        old_strategy = self.sample_strategy
        old_ratio = self.hard_negative_ratio
        
        self.sample_strategy = strategy
        if hard_ratio is not None:
            self.hard_negative_ratio = hard_ratio
        
        # 只有在策略或比例真正改变时才重新计算
        if old_strategy != strategy or (hard_ratio is not None and abs(old_ratio - hard_ratio) > 0.01):
            print(f"更新采样策略: {old_strategy} -> {strategy}, 困难负样本比例: {old_ratio:.2f} -> {hard_ratio:.2f}")
            # 重新计算所有负样本索引
            _, self.neg_indices = self._precompute_pos_neg_indices()
        
    def refresh_negative_samples(self):
        """刷新负样本（用于每个epoch开始时的随机性）"""
        # 验证集不需要刷新
        if self.is_validation:
            return
        
        print(f"刷新负样本 (策略: {self.sample_strategy}, 困难比例: {self.hard_negative_ratio:.2f})")
        _, self.neg_indices = self._precompute_pos_neg_indices()
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """返回anchor及其正负样本的索引"""
        if self.sample_strategy == 'pure_ae':
            return {'anchor_idx':idx}
        else:
            return {
                'anchor_idx': idx,
                'pos_indices': self.pos_indices[idx],
                'neg_indices': self.neg_indices[idx]
            }


def custom_collate_fn(batch, dataset):
    """
    自定义的collate函数，收集batch中所有需要的向量
    """
    device = dataset.vectors.device
    
    # 收集所有anchor索引
    anchor_indices = [item['anchor_idx'] for item in batch]
    
    if dataset.sample_strategy == 'pure_ae':
        all_vectors = dataset.normalize_vectors[anchor_indices]                       # 这里需要选择是否需要归一化向量
        return {'vectors': all_vectors}
    
    if dataset.is_validation:
        # 验证集：需要同时处理验证集向量和训练集向量
        # anchor来自验证集，正负样本来自训练集
        
        # 收集验证集的anchor向量
        # val_vectors = dataset.vectors[anchor_indices]          # 未归一向量参与计算
        val_vectors = dataset.normalize_vectors[anchor_indices] # 归一向量参与计算
        
        # 收集训练集中需要的索引（正负样本）
        train_indices_set = set()
        for item in batch:
            train_indices_set.update(item['pos_indices'].tolist())
            train_indices_set.update(item['neg_indices'].tolist())
        
        train_indices = sorted(list(train_indices_set))
        
        # 获取训练集向量
        # train_vectors = dataset.train_vectors[train_indices]      # 未归一向量参与计算
        train_vectors = dataset.normalize_train_vectors[train_indices]  # 归一向量参与计算
        
        # 合并向量：先放验证集anchor，再放训练集向量
        all_vectors = torch.cat([val_vectors, train_vectors], dim=0)
        
        # 创建索引映射
        # anchor索引：0到batch_size-1
        batch_anchor_indices = torch.arange(len(anchor_indices), dtype=torch.long)
        
        # 训练集索引映射：原始索引 -> batch内索引
        train_idx_map = {orig_idx: batch_idx + len(anchor_indices) 
                         for batch_idx, orig_idx in enumerate(train_indices)}
        
        # 构建batch内的正负样本索引
        batch_pos_indices = []
        batch_neg_indices = []
        
        for item in batch:
            pos_batch_idx = torch.tensor([train_idx_map[idx.item()] 
                                          for idx in item['pos_indices']], dtype=torch.long)
            neg_batch_idx = torch.tensor([train_idx_map[idx.item()] 
                                          for idx in item['neg_indices']], dtype=torch.long)
            
            batch_pos_indices.append(pos_batch_idx)
            batch_neg_indices.append(neg_batch_idx)
        
        batch_pos_indices = torch.stack(batch_pos_indices)
        batch_neg_indices = torch.stack(batch_neg_indices)
        
    else:
        # 训练集：所有向量来自同一个集合
        # 收集所有需要的索引（去重）
        all_indices = set(anchor_indices)
        for item in batch:
            all_indices.update(item['pos_indices'].tolist())
            all_indices.update(item['neg_indices'].tolist())
        
        all_indices = sorted(list(all_indices))
        
        # 创建索引映射：原始索引 -> batch内索引
        idx_map = {orig_idx: batch_idx for batch_idx, orig_idx in enumerate(all_indices)}
        
        # 获取所有需要的向量
        # all_vectors = dataset.vectors[all_indices]             # 未归一向量参与计算
        all_vectors = dataset.normalize_vectors[all_indices]  # 归一向量参与计算

        # 构建batch内的索引
        batch_anchor_indices = torch.tensor([idx_map[idx] for idx in anchor_indices], dtype=torch.long)
        
        batch_pos_indices = []
        batch_neg_indices = []
        
        for item in batch:
            pos_batch_idx = torch.tensor([idx_map[idx.item()] 
                                          for idx in item['pos_indices']], dtype=torch.long)
            neg_batch_idx = torch.tensor([idx_map[idx.item()] 
                                          for idx in item['neg_indices']], dtype=torch.long)
            
            batch_pos_indices.append(pos_batch_idx)
            batch_neg_indices.append(neg_batch_idx)
        
        batch_pos_indices = torch.stack(batch_pos_indices)
        batch_neg_indices = torch.stack(batch_neg_indices)
    
    return {
        'vectors': all_vectors,
        'anchor_indices': batch_anchor_indices,
        'pos_indices': batch_pos_indices,
        'neg_indices': batch_neg_indices,
        'is_validation': dataset.is_validation
    }


def create_data_loaders(data_path, args, val_split=0.1):
    """
    创建数据加载器 - 新版本
    """
    # 加载数据
    vector_dict = load_and_preprocess_data(data_path, args.dataset_type, args.num_samples)
    
    if vector_dict is None:
        raise ValueError("数据集名称有误")
    
    # 获取所有ID并随机打乱
    all_ids = list(vector_dict.keys())
    n_samples = len(all_ids)
    
    random.shuffle(all_ids)
    
    # 划分训练集和验证集
    split_point = int(n_samples * (1 - val_split))
    train_ids = all_ids[:split_point]
    val_ids = all_ids[split_point:]
    
    # 构建向量矩阵
    train_vectors = torch.stack([vector_dict[id_] for id_ in train_ids])
    val_vectors = torch.stack([vector_dict[id_] for id_ in val_ids])
    
    # 创建训练集数据集
    print("创建训练集...")
    # 对于渐进式策略，初始使用随机采样
    if args.sample_strategy == 'progressive':
        initial_strategy = 'random'
        initial_hard_ratio = 0.0
    else:
        initial_strategy = args.sample_strategy
        initial_hard_ratio = args.hard_negative_ratio
    
    train_dataset = VectorDataset(
        train_vectors,
        train_ids,
        args=args,
        k_neighbors=args.k_neighbors,
        neg_sample_ratio=args.neg_sample_ratio,
        hard_negative_ratio=args.hard_negative_ratio,  # 目标比例
        hard_negative_range=args.hard_negative_range,
        sample_strategy=initial_strategy,
        initial_hard_ratio=initial_hard_ratio  # 初始比例
    )
    
    # 创建验证集数据集（使用随机采样，在训练集中寻找近邻）
    print("创建验证集...")
    val_dataset = VectorDataset(
        val_vectors,
        val_ids,
        args=args,
        k_neighbors=args.k_neighbors,
        neg_sample_ratio=args.neg_sample_ratio,
        hard_negative_ratio=0.0,  # 验证集使用随机采样
        hard_negative_range=args.hard_negative_range,
        sample_strategy=initial_strategy if args.sample_strategy !="hard" else "random", 
        train_vectors=train_vectors,  # 提供训练集向量
        train_ids=train_ids,  # 提供训练集ID
        is_validation=True  # 标记为验证集
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=lambda batch: custom_collate_fn(batch, train_dataset)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=lambda batch: custom_collate_fn(batch, val_dataset)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"k近邻数: {args.k_neighbors}")
    print(f"负样本比例: {args.neg_sample_ratio}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_vectors.numpy(),val_vectors.numpy()


# 保留原有的辅助函数
def compute_knn_order(embeddings, k, distance_metric, ids=None):
    """计算k近邻顺序"""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    n_samples = embeddings.shape[0]
    
    if ids is None:
        ids = list(range(n_samples))
    if distance_metric=='euclidean':
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    
    distances, indices = index.search(embeddings.astype(np.float32), min(k + 1, n_samples))
    
    knn_order = {}
    for i, id_ in enumerate(ids):
        neighbors = [(ids[indices[i, j]], float(distances[i, j])) 
                    for j in range(1, min(k + 1, n_samples))]
        knn_order[id_] = neighbors
    
    return knn_order


def load_and_preprocess_data(data_path, dataset_type, num_samples):
    """加载和预处理数据"""
    if dataset_type == "CelebA":
        print(f"加载数据集:{dataset_type}")
        file_path = os.path.join(data_path, dataset_type, "raw_embeddings_withname.pkl")
        with open(file_path, "rb") as file:
            raw_embeddings_withname = pickle.load(file)
        
        all_embeddings = []
        all_original_ids = []
        all_original_jpgname = []
        
        for original_id, jpg_dict in raw_embeddings_withname.items():
            for jpg_name, embedding in jpg_dict.items():
                all_embeddings.append(embedding)
                all_original_ids.append(original_id)
                all_original_jpgname.append(jpg_name)
        
        total_samples = len(all_embeddings)
        if num_samples > total_samples:
            print(f"Warning: 请求的样本数量({num_samples})超过了总样本数量({total_samples})")
            num_samples = total_samples
        
        selected_indices = random.sample(range(total_samples), num_samples)
        
        selected_embeddings = {}
        for new_idx, original_idx in enumerate(selected_indices):
            selected_embeddings[new_idx] = torch.tensor(all_embeddings[original_idx], dtype=torch.float32)
        
        return selected_embeddings
    
    else:
        return None