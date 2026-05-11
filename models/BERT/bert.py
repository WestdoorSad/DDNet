import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BERT.transformer import TransformerBlock

class NextSignalPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

def construct_gcm(signal, grid_size=20):
    import numpy as np
    def normalize_signal(signal):
        avg_power = np.mean(np.abs(signal) ** 2)
        normalized_signal = signal / np.sqrt(avg_power)
        return normalized_signal

    signal = normalize_signal(signal)
    real_part = signal[0]
    imag_part = signal[1]

    real_part_seg = (real_part.max() - real_part.min()) / (grid_size - 1)
    imag_part_seg = (imag_part.max() - imag_part.min()) / (grid_size - 1)


    num_symbols = signal.shape[1]
    gcm = np.zeros((grid_size, grid_size))

    for i in range(num_symbols):
        real_pos = (signal[0, i] - real_part.min()) / real_part_seg
        imag_pos = (signal[1, i] - imag_part.min()) / imag_part_seg
        gcm[int(real_pos), int(imag_pos)] += 1

    gcm = gcm.astype(np.float32)
    gcm = gcm / num_symbols
    gcm = gcm.ravel()

    return gcm


class SPAtt_1D(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # 将 Conv2d 改为 Conv1d，输入通道数为 2，输出通道数为 1
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding="same", bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状为 [b, c, w]
        # 计算平均值和最大值，保持维度
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [b, 1, w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [b, 1, w]

        # 将 avg_out 和 max_out 在通道维度上拼接
        out = torch.cat([avg_out, max_out], dim=1)  # [b, 2, w]

        # 应用 Conv1d
        out = self.conv(out)  # [b, 1, w]

        # 应用 Sigmoid 激活函数
        out = self.sigmoid(out)

        # 返回加权后的输入
        return out * x  # [b, 1, w] * [b, c, w] -> [b, c, w]


class CHAtt_1D(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # 将 AdaptiveAvgPool2d 和 AdaptiveMaxPool2d 改为 AdaptiveAvgPool1d 和 AdaptiveMaxPool1d
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 输出形状: [b, c, 1]
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 输出形状: [b, c, 1]

        # 将 Conv2d 改为 Conv1d
        self.fc = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),  # 1x1 卷积
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),  # 1x1 卷积
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状: [b, c, w]
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))  # [b, c, 1] -> [b, c, 1]
        max_out = self.fc(self.max_pool(x))  # [b, c, 1] -> [b, c, 1]

        # 将 avg_out 和 max_out 相加
        out = avg_out + max_out  # [b, c, 1]

        # 应用 Sigmoid 激活函数
        out = self.sigmoid(out)  # [b, c, 1]

        # 返回加权后的输入
        return out * x  # [b, c, 1] * [b, c, w] -> [b, c, w]

class EMB_head(nn.Module):
    def __init__(self, hidden=64, drop_out=0.1):
        super().__init__()
        assert  hidden % 4 == 0
        self.cnn1 = nn.Sequential(
            nn.Conv1d(2, hidden // 8, 3, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 8, num_groups=hidden // 8),
            nn.GELU(),)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(2, hidden // 4, 5, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 4, num_groups=hidden // 4),
            nn.GELU(),)
        self.cnn3 = nn.Sequential(
            nn.Conv1d(2, hidden // 8, 7, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 8, num_groups=hidden // 8),
            nn.GELU(),)
        self.cnn4 = nn.Sequential(
            nn.Conv1d(2, hidden // 4, 9, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 4, num_groups=hidden // 4),
            nn.GELU(),)
        self.cnn5 = nn.Sequential(
            nn.Conv1d(2, hidden // 8, 11, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 8, num_groups=hidden // 8),
            nn.GELU(),)
        self.cnn6 = nn.Sequential(
            nn.Conv1d(2, hidden // 8, 13, stride=1, padding="same"),
            nn.GroupNorm(num_channels=hidden // 8, num_groups=hidden // 8),
            nn.GELU(),)
        self.dp = nn.Dropout(drop_out)
        self.out = nn.Sequential(
            CHAtt_1D(hidden),
            SPAtt_1D(hidden),
            nn.Conv1d(hidden, hidden, 1, padding="same", groups=8),
            nn.GroupNorm(num_channels=hidden, num_groups=hidden),
            nn.GELU(),
            nn.MaxPool1d(2),
        )


    def forward(self, x):

        x = torch.squeeze(x, dim=1)

        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x4 = self.cnn4(x)
        x5 = self.cnn5(x)
        x6 = self.cnn6(x)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = self.dp(x)
        x = self.out(x).transpose(1, 2)

        return x

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = EMB_head(hidden)
        # self.embedding = EMB_head(hidden)

        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, hidden))

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden) for _ in range(n_layers)])

        self.dp = nn.Dropout(dropout)



    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)

        x = self.embedding(x)
        x = x + self.position_embeddings

        x = self.dp(x)

        seq_len = x.shape[1]
        mask = torch.zeros((x.shape[0], 1, seq_len, seq_len)).to(x.device)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def pre_forward(self, x, mask=None, **kwargs):
        if mask is None:
            seq_len = x.shape[-1] // 2
            mask = torch.zeros((x.shape[0], 1, seq_len, seq_len)).to(x.device)

        x = self.embedding(x)
        x = x + self.position_embeddings

        x = self.dp(x)

        out = []

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
            out.append(x)

        return x, out
