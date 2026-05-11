import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA
import random


@torch.no_grad()
def concat_all_gather(x):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(x)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, x, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def contrastive_loss(q, k, T=0.07):
    # normalize
    q = nn.functional.normalize(q.reshape(q.shape[0], -1), dim=1)
    k = nn.functional.normalize(k.reshape(q.shape[0], -1), dim=1)
    # Einstein sum is more intuitive
    logits = torch.einsum('nc,mc->nm', [q, k]) / T
    N = logits.shape[0]  # batch size per GPU
    labels = (torch.arange(N, dtype=torch.long)).cuda()
    return nn.CrossEntropyLoss()(logits, labels) * (2 * T)

class Data2Vec(nn.Module):

    def __init__(self, encoder, embed_dim=48, **kwargs):
        super(Data2Vec, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.ema = EMA(self.encoder, 0.99)  # EMA acts as the teacher
        self.regression_head = nn.Linear(self.embed_dim, self.embed_dim)

        self.contrastive_reg_head_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.contrastive_reg_head_q = nn.Linear(self.embed_dim, self.embed_dim)

        self.ema_decay = 0.99
        self.ema_end_decay = 0.9999
        self.ema_anneal_end_step = 3000

        self.mask_ratio = 0.5

        self.cls = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.AvgPool1d(2),
            nn.Flatten(),
            nn.Linear(self.embed_dim * 8, 11),
        )

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)



    def forward(self, src, trg, mask=None, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        # model forward in online mode (student)
        seq_len = src.shape[-1] // 2
        mask = torch.zeros(src.shape[0], seq_len)

        batch_size = src.shape[0]
        for i in range(batch_size):
            num_to_mask = int(seq_len * self.mask_ratio) 
            random_mask_indices_in_sample = random.sample(list(range(seq_len)), num_to_mask)
            mask[i][random_mask_indices_in_sample] = 1

        mask_use = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1).to(src.device)
        x, _ = self.encoder.pre_forward(src, mask=mask_use, **kwargs)  # fetch the last layer outputs

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            _, y = self.ema.model.pre_forward(trg)  # fetch the last transformer layers outputs
            y = y[-2:]  # take the last k transformer layers
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)

        x = x[mask == 1]
        y = y[mask == 1]

        x = self.regression_head(x)

        return x, y
