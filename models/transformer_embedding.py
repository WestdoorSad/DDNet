from torch import nn
import torch
import numpy as np

# 定义Transformer模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_size, num_layers=2, num_heads=4, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.src_mask = None
        self.enc = nn.Conv1d(2, 32, 3,1, padding=1)
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = torch.squeeze(src, dim=1)
        src = self.enc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        output = output.view(output.size(0), -1)
        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, input_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 生成正弦用的div_term，对于维度为9的输入，我们需要5个正弦值
        div_term_even = torch.exp(torch.arange(0, input_size, 2).float() * (-np.log(10000.0) / input_size))
        # 生成余弦用的div_term，对于维度为9的输入，我们需要4个余弦值
        div_term_odd = torch.exp(torch.arange(1, input_size, 2).float() * (-np.log(10000.0) / input_size))

        # 正弦赋值，对于维度为9的输入，我们应该赋值给索引0, 2, 4, 6, 8
        pe[:, 0::2] = torch.sin(position * div_term_even.unsqueeze(0))

        # 余弦赋值，对于维度为9的输入，我们应该赋值给索引1, 3, 5, 7
        pe[:, 1::2] = torch.cos(position * div_term_odd.unsqueeze(0))

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == "__main__":
    model = SimpleTransformer(128)
    inx = torch.zeros(2, 2, 128)
    outy = model(inx)