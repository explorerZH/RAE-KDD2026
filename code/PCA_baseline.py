from data_utils import create_data_loaders, compute_knn_order
from config import get_args
import wandb
import random
import numpy as np
import torch
from datetime import datetime
from config import get_args
from data_utils import create_data_loaders, compute_knn_order
import faiss
from sklearn.decomposition import PCA

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_vectors(vectors):
    """归一化向量"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-8)

def compute_knn_order(embeddings, k, distance_metric='cosine'):
    """
    计算k近邻顺序（同一集合内）
    直接返回索引和距离，不使用ID系统
    """
    n_samples = embeddings.shape[0]
    
    # 根据距离度量选择FAISS索引
    if distance_metric == 'cosine':
        # 对于余弦相似度，先归一化再使用内积
        embeddings = normalize_vectors(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    elif distance_metric == 'inner_product':
        index = faiss.IndexFlatIP(embeddings.shape[1])
    elif distance_metric == 'euclidean':
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        raise ValueError(f"不支持的距离度量: {distance_metric}")
    
    index.add(embeddings.astype(np.float32))
    
    # 搜索k+1个最近邻（包括自己）
    distances, indices = index.search(embeddings.astype(np.float32), min(k + 1, n_samples))
    
    knn_order = {}
    for i in range(n_samples):
        # 排除自己（第一个最近邻）
        neighbors = [(int(indices[i, j]), float(distances[i, j])) 
                    for j in range(1, min(k + 1, n_samples))]
        knn_order[i] = neighbors
    
    return knn_order

def compute_knn_cross_set(query_vectors, base_vectors, k, distance_metric='cosine'):
    """
    计算查询集在基准集中的k近邻
    直接返回索引和距离，不使用ID系统
    """
    # 根据距离度量选择FAISS索引
    if distance_metric == 'cosine':
        # 对于余弦相似度，先归一化再使用内积
        query_vectors = normalize_vectors(query_vectors)
        base_vectors = normalize_vectors(base_vectors)
        index = faiss.IndexFlatIP(base_vectors.shape[1])
    elif distance_metric == 'inner_product':
        index = faiss.IndexFlatIP(base_vectors.shape[1])
    elif distance_metric == 'euclidean':
        index = faiss.IndexFlatL2(base_vectors.shape[1])
    else:
        raise ValueError(f"不支持的距离度量: {distance_metric}")
    
    index.add(base_vectors.astype(np.float32))
    
    distances, indices = index.search(query_vectors.astype(np.float32), k)
    
    # 构建k近邻字典
    knn_order = {}
    for i in range(query_vectors.shape[0]):
        neighbors = [(int(indices[i, j]), float(distances[i, j])) 
                    for j in range(k)]
        knn_order[i] = neighbors
    
    return knn_order

def calculate_accuracy(base_order, reduction_order, topK_num_list):
    """计算降维前后k近邻的重合准确度"""
    entity_ids = list(base_order.keys())
    
    similarity_dict = {}
    for k in topK_num_list:
        total_similarity = 0
        for id in entity_ids:
            # 提取前k个近邻的索引
            set_a = set(item[0] for item in base_order[id][:k])
            set_b = set(item[0] for item in reduction_order[id][:k])
            
            # 计算交集
            correct = len(set_b.intersection(set_a))
            similarity = correct / k
            total_similarity += similarity
        
        average_similarity = total_similarity / len(entity_ids)
        similarity_dict[f"top{k}"] = average_similarity
    
    return similarity_dict

def evaluate_pca_knn_preservation(train_vectors, val_vectors, args):
    """评估PCA降维后的k近邻保持度"""
    
    print(f"\n开始PCA降维实验...")
    print(f"训练集大小: {train_vectors.shape}")
    print(f"验证集大小: {val_vectors.shape}")
    print(f"原始维度: {train_vectors.shape[1]}")
    print(f"目标维度: {args.output_dim}")
    print(f"距离度量: {args.distance_metric}")
    print(f"评估的k值: {args.topk_eval}")
    
    # 1. 训练PCA模型（使用训练集）
    print("\n训练PCA模型...")
    pca = PCA(n_components=args.output_dim, random_state=args.seed)
    unnormalized_train_vectors = train_vectors
    unnormalized_val_vectors = val_vectors
    # 计算前是否归一
    #train_vectors = normalize_vectors(train_vectors)
    pca.fit(train_vectors)
    #val_vectors = normalize_vectors(val_vectors)
    # 输出解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"前{args.output_dim}个主成分解释的方差比例: {explained_variance_ratio:.4f}")
    
    # 2. 对训练集和验证集进行降维
    train_reduced = pca.transform(train_vectors)
    val_reduced = pca.transform(val_vectors)
    # 降维后是否归一
    # train_reduced = normalize_vectors(train_reduced)
    # val_reduced = normalize_vectors(val_reduced)

    results = {}
    
    # 3. 计算训练集内部的k近邻保持度
    print("\n计算训练集内部的k近邻保持度...")
    max_k = max(args.topk_eval)
    
    # 计算原始空间的k近邻
    train_original_knn = compute_knn_order(
        unnormalized_train_vectors, max_k, args.distance_metric
    )
    
    # 计算降维空间的k近邻
    train_reduced_knn = compute_knn_order(
        train_reduced, max_k, args.distance_metric
    )
    
    # 计算准确度
    train_accuracy = calculate_accuracy(
        train_original_knn,
        train_reduced_knn,
        args.topk_eval
    )
    
    # 4. 计算验证集在训练集中的k近邻保持度
    print("计算验证集在训练集中的k近邻保持度...")
    
    # 计算原始空间中验证集在训练集中的k近邻
    val_original_knn = compute_knn_cross_set(
        unnormalized_val_vectors, unnormalized_train_vectors, max_k, args.distance_metric
    )
    
    # 计算降维空间中验证集在训练集中的k近邻
    val_reduced_knn = compute_knn_cross_set(
        val_reduced, train_reduced, max_k, args.distance_metric
    )
    
    # 计算准确度
    val_accuracy = calculate_accuracy(
        val_original_knn,
        val_reduced_knn,
        args.topk_eval
    )
    
    results['train'] = train_accuracy
    results['val'] = val_accuracy
    results['explained_variance_ratio'] = explained_variance_ratio
    
    return results

def print_results(results):
    """打印实验结果"""
    print("\n" + "="*60)
    print("实验结果")
    print("="*60)
    
    print(f"\n解释方差比例: {results['explained_variance_ratio']:.4f}")
    
    print("\n训练集k近邻保持度:")
    for k, acc in results['train'].items():
        print(f"  {k}: {acc:.4f}")
    
    print("\n验证集k近邻保持度:")
    for k, acc in results['val'].items():
        print(f"  {k}: {acc:.4f}")
    

def main():
    args = get_args()
    set_seed(args.seed)
    
    # 初始化wandb
    if args.use_wandb:
        
        # 正式实验时请在此处写明本组实验所探究的内容，方便官网浏览
        wandb.init(
            project="PCA_baseline",
            entity=args.wandb_entity,
            name=f"PCA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",       
            config=vars(args)
        )
    
    # 加载数据
    _, _, _, _, train_vectors, val_vectors = create_data_loaders(
        args.data_path, args
    )
    
    # 评估PCA降维的k近邻保持度
    results = evaluate_pca_knn_preservation(
        train_vectors, val_vectors, 
        args
    )
    
    # 打印结果
    print_results(results)
    
    # 记录到wandb
    if args.use_wandb:
        # 记录详细的k值结果
        for split in ['train', 'val']:
            for k, acc in results[split].items():
                wandb.log({f"{split}_{k}_accuracy": acc})
        
        wandb.finish()
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()