
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch.nn.functional as F
import torch

#############################
# MULTI HEAD SELF ATTENTION #
#############################
class SelfAttention(nn.Module):
    def __init__(self, embeddings_size=768, heads=1, attn_dropout=0.1):
        super(SelfAttention, self).__init__()

        self.embeddings_size = embeddings_size
        self.heads = heads
        self.head_dim = embeddings_size//heads

        self.query = nn.Linear(embeddings_size, embeddings_size)
        self.key = nn.Linear(embeddings_size, embeddings_size)
        self.val = nn.Linear(embeddings_size, embeddings_size)

        self.fcc_layer = nn.Linear(embeddings_size, embeddings_size)

        self.drop_layer = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v):
        ##num elements in batch
        batch_size = q.size()[0]
        ##num tokens
        seq_len = q.size()[1]

        q = self.query(q).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)
        k = self.query(k).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)
        v = self.query(v).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,2)

        attn_weights = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        attn_weights = self.drop_layer(F.softmax(attn_weights, dim=-1))
        output = torch.matmul(attn_weights, v)

        original_dim_output = output.transpose(1,2).contiguous().view(batch_size, -1, self.embeddings_size)
        transformed_output = self.fcc_layer(original_dim_output)

        

        return transformed_output