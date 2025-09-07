import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Yann LeCun的对比损失函数 - 修正版本
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels, sampled_indices, sampled_labels):
        """
        Args:
            embeddings: 降维后的嵌入向量 [batch_size, embedding_dim]
            labels: 全量k近邻标签矩阵 [batch_size, num_samples]
            sampled_indices: 采样的样本索引 [batch_size, num_sampled]
            sampled_labels: 采样后的标签 [batch_size, num_sampled]
        """
        batch_size = embeddings.size(0)
        
        # 归一化嵌入向量
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 获取采样的嵌入向量
        sampled_embeddings = embeddings[sampled_indices.flatten()].view(
            batch_size, sampled_indices.size(1), -1
        )
        sampled_embeddings_norm = F.normalize(sampled_embeddings, p=2, dim=2)
        
        # 计算余弦相似度（归一化后的内积）
        similarities = torch.matmul(embeddings_norm.unsqueeze(1), sampled_embeddings_norm.transpose(-2, -1)).squeeze(1)
        
        # 将相似度转换为距离（距离 = 1 - 相似度）
        distances = 1 - similarities
        
        # 正样本对损失：希望距离接近0
        pos_loss = sampled_labels * distances.pow(2)
        
        # 负样本对损失：希望距离大于margin
        neg_loss = (1 - sampled_labels) * F.relu(self.margin - distances).pow(2)
        
        # 总损失
        loss = 0.5 * (pos_loss + neg_loss)
        
        return loss.mean()


class TripletLoss(nn.Module):
    """
    三元组损失函数 - 修正版本
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels, sampled_indices, sampled_labels):
        """
        Args:
            embeddings: 降维后的嵌入向量 [batch_size, embedding_dim]
            labels: 全量k近邻标签矩阵 [batch_size, num_samples]
            sampled_indices: 采样的样本索引 [batch_size, num_sampled]
            sampled_labels: 采样后的标签 [batch_size, num_sampled]
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 归一化嵌入向量
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 获取采样的嵌入向量
        sampled_embeddings = embeddings[sampled_indices.flatten()].view(
            batch_size, sampled_indices.size(1), -1
        )
        sampled_embeddings_norm = F.normalize(sampled_embeddings, p=2, dim=2)
        
        # 计算余弦相似度
        similarities = torch.matmul(embeddings_norm.unsqueeze(1), sampled_embeddings_norm.transpose(-2, -1)).squeeze(1)
        
        # 将相似度转换为距离
        distances = 1 - similarities
        
        # 对每个anchor找到正样本和负样本
        loss = 0
        valid_triplets = 0
        
        for i in range(batch_size):
            # 获取正样本和负样本的索引
            pos_indices = (sampled_labels[i] == 1).nonzero(as_tuple=True)[0]
            neg_indices = (sampled_labels[i] == 0).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # 获取正样本和负样本的距离
                pos_distances = distances[i, pos_indices]
                neg_distances = distances[i, neg_indices]
                
                # 计算所有正负样本对的三元组损失
                pos_distances_expanded = pos_distances.unsqueeze(1)
                neg_distances_expanded = neg_distances.unsqueeze(0)
                
                triplet_loss = F.relu(pos_distances_expanded - neg_distances_expanded + self.margin)
                loss += triplet_loss.mean()
                valid_triplets += 1
        
        if valid_triplets > 0:
            loss = loss / valid_triplets
        
        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数 - 向量化版本
    分子：所有正样本与anchor的相似度的exp之和
    分母：所有样本（正样本+负样本）与anchor的相似度的exp之和
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, all_embeddings, anchor_indices, pos_indices, neg_indices):
        """
        完全向量化的InfoNCE损失计算
        
        Args:
            all_embeddings: 批次中所有向量的嵌入 [total_vectors, embedding_dim]
            anchor_indices: anchor在all_embeddings中的索引 [batch_size]
            pos_indices: 正样本索引 [batch_size, k]
            neg_indices: 负样本索引 [batch_size, num_neg]
        
        Returns:
            loss: InfoNCE损失值
        """
        batch_size = anchor_indices.size(0)
        k = pos_indices.size(1)
        num_neg = neg_indices.size(1)
        
        # 归一化所有嵌入（用于计算余弦相似度）
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
        
        # 获取anchor嵌入
        anchor_emb = all_embeddings_norm[anchor_indices]  # [batch_size, embedding_dim]
        
        # 获取正负样本嵌入
        pos_emb = all_embeddings_norm[pos_indices]  # [batch_size, k, embedding_dim]
        neg_emb = all_embeddings_norm[neg_indices]  # [batch_size, num_neg, embedding_dim]
        
        # 计算相似度（内积）
        # Anchor与正样本的相似度
        pos_sim = torch.bmm(pos_emb, anchor_emb.unsqueeze(2)).squeeze(2) / self.temperature
        # [batch_size, k]
        
        # Anchor与负样本的相似度
        neg_sim = torch.bmm(neg_emb, anchor_emb.unsqueeze(2)).squeeze(2) / self.temperature
        # [batch_size, num_neg]
        
        # 计算分子：所有正样本的exp之和
        pos_exp_sum = torch.exp(pos_sim).sum(dim=1)  # [batch_size]
        
        # 计算分母：所有样本（正样本+负样本）的exp之和

        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, k + num_neg]
        all_exp_sum = torch.exp(all_sim).sum(dim=1)  # [batch_size]
        
        # InfoNCE损失：-log(正样本exp之和 / 所有样本exp之和)
        loss = -torch.log(pos_exp_sum / all_exp_sum)
        
        return loss.mean()

class K_preserving_loss(nn.Module):
    """
    K近邻保持损失函数
    包含：
    1. 重构损失：anchor的重构向量与原始向量的MSE
    2. 近邻保持损失：k近邻的重构向量与anchor原始向量的加权MSE
    3. 负样本间隔损失：负样本与anchor的距离应大于margin（可选）
    """
    
    def __init__(self,
                 k=5,
                 neighbor_weights=None,
                 use_negative_loss=False,
                 negative_margin=1.0,
                 negative_weight=0.1):
        """
        Args:
            k (int): 近邻数量
            neighbor_weights (list or None): 每个近邻的权重列表，长度应为k
                                            如果为None，则自动生成递减权重
            use_negative_loss (bool): 是否使用负样本间隔损失
            negative_margin (float): 负样本的最小距离间隔
            negative_weight (float): 负样本损失的权重
        """
        super().__init__()
        
        self.k = k
        
        if neighbor_weights is None:
            # 自动生成递减权重：最近的邻居权重最大
            # 例如 k=5 时: [1.0, 0.8, 0.6, 0.4, 0.2]
            self.neighbor_weights = torch.linspace(1.0, 0.2, k)
        else:
            assert len(neighbor_weights) == k, f"权重列表长度必须等于k={k}"
            self.neighbor_weights = torch.tensor(neighbor_weights, dtype=torch.float32)
        
        self.use_negative_loss = use_negative_loss
        self.negative_margin = negative_margin
        self.negative_weight = negative_weight
        
    def forward(self, 
                all_reconstructed, 
                all_vectors, 
                anchor_indices, 
                pos_indices, 
                neg_indices=None):
        """
        计算K近邻保持损失
        
        Args:
            all_reconstructed: [total_size, embedding_dim] - 所有向量的重构结果
            all_vectors: [total_size, embedding_dim] - 所有原始向量
            anchor_indices: [batch_size] - anchor向量的索引
            pos_indices: [batch_size, k] - 每个anchor对应的k个正样本索引
            neg_indices: [batch_size, num_neg] - 每个anchor对应的负样本索引（可选）
            
        Returns:
            loss: 标量损失值，可直接用于backward
        """
        device = all_reconstructed.device
        batch_size = anchor_indices.shape[0]
        
        # 将权重移到正确的设备
        weights = self.neighbor_weights.to(device)
        
        # 获取anchor的原始向量和重构向量
        anchor_raw_vectors = all_vectors[anchor_indices]  # [batch_size, embedding_dim]
        anchor_recon_vectors = all_reconstructed[anchor_indices]  # [batch_size, embedding_dim]
        
        # 获取正样本的重构向量
        # 需要reshape pos_indices来正确索引
        pos_indices_flat = pos_indices.reshape(-1)  # [batch_size * k]
        pos_emb_flat = all_reconstructed[pos_indices_flat]  # [batch_size * k, embedding_dim]
        pos_emb = pos_emb_flat.reshape(batch_size, self.k, -1)  # [batch_size, k, embedding_dim]
        
        # 1. 重构损失 - anchor重构向量与原始向量的MSE
        recon_loss = F.mse_loss(anchor_recon_vectors, anchor_raw_vectors)
        
        # 2. 近邻保持损失 - k近邻重构向量与anchor原始向量的加权MSE
        anchor_raw_expanded = anchor_raw_vectors.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # 计算每个近邻与anchor原始向量的欧氏距离平方
        neighbor_distances = torch.sum((pos_emb - anchor_raw_expanded) ** 2, dim=-1)  # [batch_size, k]
        
        # 应用每个近邻的权重
        weights_expanded = weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, k]
        weighted_distances = neighbor_distances * weights_expanded
        neighbor_loss = weighted_distances.mean()
        
        # 3. 负样本间隔损失（可选）
        negative_loss = torch.tensor(0.0, device=device)
        if self.use_negative_loss and neg_indices is not None:
            # 获取负样本的重构向量
            num_neg = neg_indices.shape[1]
            neg_indices_flat = neg_indices.reshape(-1)  # [batch_size * num_neg]
            neg_emb_flat = all_reconstructed[neg_indices_flat]  # [batch_size * num_neg, embedding_dim]
            neg_emb = neg_emb_flat.reshape(batch_size, num_neg, -1)  # [batch_size, num_neg, embedding_dim]
            
            # 计算负样本与anchor原始向量的欧氏距离
            anchor_raw_neg = anchor_raw_vectors.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            neg_distances = torch.sqrt(torch.sum((neg_emb - anchor_raw_neg) ** 2, dim=-1) + 1e-8)  # [batch_size, num_neg]
            
            # Hinge loss: max(0, margin - distance)
            # 我们希望距离大于margin，所以当distance < margin时会产生损失
            margin_violations = F.relu(self.negative_margin - neg_distances)
            negative_loss = self.negative_weight * margin_violations.mean()
        
        # 计算总损失
        total_loss = recon_loss + neighbor_loss + negative_loss
        
        return total_loss
         

