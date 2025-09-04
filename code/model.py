import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    """
    基于MLP的AutoEncoder模型
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256], 
                 activation='relu', dropout=0.1):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()

        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim

        if hidden_dims:  # 如果有隐藏层
            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.activation,
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
        # 添加最后的降维层
            encoder_layers.append(nn.Linear(prev_dim, output_dim))
        else:  # 直接映射，无隐藏层
            encoder_layers.append(nn.Linear(input_dim, output_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # 构建解码器 (镜像结构)
        decoder_layers = []
        prev_dim = output_dim

        if hidden_dims:  # 如果有隐藏层
            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self.activation,
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            # 添加最后的重构层
            decoder_layers.append(nn.Linear(prev_dim, input_dim))
        else:  # 直接映射，无隐藏层
            decoder_layers.append(nn.Linear(output_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)
        
    
    def encode(self, x):
        """编码器：将高维向量降维"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码器：从低维向量重构高维向量"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed
    
    def get_embeddings(self, x):
        """获取降维后的嵌入向量"""
        with torch.no_grad():
            return self.encode(x)
        