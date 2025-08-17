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