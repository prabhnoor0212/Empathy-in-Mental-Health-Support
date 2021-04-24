
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch.nn.functional as F
import torch
from config import _max_tokenizer_len

#############################
#    Dense Synthesizer      #
#############################
class DenseSynthesizer(nn.Module):
    """ Class for Dense Syntehsizer implementation
    """
    def __init__(self, embeddings_size=768, n_tokens=_max_tokenizer_len, syn_dropout=0.1):
        super(DenseSynthesizer, self).__init__()
        self.embeddings_size = embeddings_size
        self.d_projection = nn.Linear(embeddings_size, embeddings_size)
        self.synth_projection = nn.Linear(embeddings_size, n_tokens)
        self.g_projection = nn.Linear(embeddings_size, embeddings_size)
        self.drop_layer = nn.Dropout(syn_dropout)
        self.final_projection = nn.Linear(embeddings_size, embeddings_size)

    def forward(self, q, v):
        ##num elements in batch
        batch_size = v.size()[0]
        ##num tokens
        seq_len = v.size()[1]

        b = self.synth_projection(torch.relu(self.d_projection(q)))
        g = self.g_projection(v)
        g.view(batch_size, seq_len, self.embeddings_size).transpose(-1,-2)

        b = self.drop_layer(F.softmax(b, dim=-1))

        synth_output = torch.matmul(b, g)
        transformed_output = self.final_projection(synth_output)
        return transformed_output

#############################
# MULTI HEAD SELF ATTENTION #
#############################
class SelfAttention(nn.Module):
    """Class for Multi-Head Attention
    """
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


#############################
#  Talking Heads ATTENTION  #
#############################

class TalkingHeadsAttention(nn.Module):
    """Class for multi-head attention
    """
    def __init__(self, embeddings_size=768, heads=12, attn_dropout=0.1):
        super(TalkingHeadsAttention, self).__init__()

        self.embeddings_size = embeddings_size
        self.heads = heads
        self.head_dim = embeddings_size//heads

        self.query = nn.Linear(embeddings_size, embeddings_size)
        self.key = nn.Linear(embeddings_size, embeddings_size)
        self.val = nn.Linear(embeddings_size, embeddings_size)

        self.logits_proj = nn.Linear(heads, heads)
        self.weights_proj = nn.Linear(heads, heads)

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

        logits = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        logits = self.logits_proj(logits.transpose(1,-1)).transpose(1,-1)
        attn_weights = self.drop_layer(F.softmax(logits, dim=-1))
        attn_weights = self.weights_proj(attn_weights.transpose(1,-1)).transpose(1,-1)

        output = torch.matmul(attn_weights, v)

        original_dim_output = output.transpose(1,2).contiguous().view(batch_size, -1, self.embeddings_size)
        transformed_output = self.fcc_layer(original_dim_output)

        

        return transformed_output