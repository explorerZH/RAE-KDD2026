import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='AutoEncoder with Contrastive Learning for Dimensionality Reduction')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, default="./data", help='输入数据路径')
    parser.add_argument('--output_dim', type=int, default=128, help='降维后的目标维度')
    parser.add_argument('--dataset_type', type=str, default='CelebA', help='数据集类型')
    parser.add_argument('--num_samples', type=int, default=10000, help='使用向量个数')
    
    # 模型相关参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128], 
                        help='AutoEncoder隐藏层维度列表')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'tanh', 'sigmoid'], help='激活函数')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--alpha', type=float, default=1, 
                        help='对比损失权重 (总损失 = (1-alpha)*重构损失 + alpha*对比损失)')
    
    # 对比学习相关参数
    parser.add_argument('--k_neighbors', type=int, default=10, help='k近邻数量')
    parser.add_argument('--loss_type', type=str, default='infonce', 
                        choices=['contrastive', 'triplet', 'infonce'], help='对比损失类型')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE温度参数')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss的margin')
    
    # 采样策略相关参数
    parser.add_argument('--sample_strategy', type=str, default='progressive',
                        choices=['random', 'hard', 'progressive'], 
                        help='采样策略：random(随机), hard(困难负样本挖掘), progressive(渐进式)')
    parser.add_argument('--neg_sample_ratio', type=int, default=4, 
                        help='负样本采样比例 (相对于正样本数量)')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.7,
                        help='困难负样本占总负样本的比例 (仅在非random策略下使用)')
    parser.add_argument('--hard_negative_range', type=int, default=30,
                        help='困难负样本的范围，即从k+1到k+h的近邻 (仅在非random策略下使用)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='使用随机采样的预热轮数 (仅在progressive策略下使用)')
    parser.add_argument('--progressive_hard_ratio', type=float, default=0.05,
                        help='渐进式策略中初始的困难负样本比例 (仅在progressive策略下使用)')
    
    # 评估相关参数
    parser.add_argument('--eval_ratio', type=float, default=0.1, 
                        help='评估时采样的实体比例')
    parser.add_argument('--topk_eval', nargs='+', type=int, default=[5,10,20,50], 
                        help='评估时的top-k值列表')
    parser.add_argument('--eval_interval', type=int, default=4, help='评估间隔')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--wandb_project', type=str, default='ae-contrastive', help='wandb项目名')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb运行entity')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.sample_strategy == 'hard':
        print(f"使用困难负样本挖掘策略:")
        print(f"  - 困难负样本比例: {args.hard_negative_ratio}")
        print(f"  - 困难负样本范围: k+1 到 k+{args.hard_negative_range}")
        print(f"  - 负样本总比例: {args.neg_sample_ratio}")
    elif args.sample_strategy == 'progressive':
        print(f"使用渐进式采样策略:")
        print(f"  - 预热轮数: {args.warmup_epochs}")
        print(f"  - 初始困难负样本比例: {args.progressive_hard_ratio}")
        print(f"  - 最终困难负样本比例: {args.hard_negative_ratio}")
        print(f"  - 困难负样本范围: k+1 到 k+{args.hard_negative_range}")
        print(f"  - 负样本总比例: {args.neg_sample_ratio}")
    else:
        print(f"使用随机负样本采样策略:")
        print(f"  - 负样本比例: {args.neg_sample_ratio}")
    
    return args