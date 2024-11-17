import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math



class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=5000):
        """Sinusoidal Positional Encoding Constructor

        Args:
            d_model (int): The number of features/embedding dimension
            max_len (int, optional): _description_. Defaults to 5000.
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        #Create a pe tensor of shape max_len, d_model
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        #Generates a tensor of position indices from 0 to max_len-1 (index)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Calculates the division terms for the sinusoidal functions.
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        #Even index sin Odd index cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #make pe tensor of shape 1,max_len,d_model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # take out (1, sequence_length, d_model)
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Token Embedding Embeddes each token (Embedder of Transformer)"""
    def __init__(self, c_in, d_model):
        """Constructor of Token Embedder of Transformer

        Args:
            c_in (_type_): the number of feature we would like to pass in for embedding
            d_model (_type_): dimension that we would like the embedder to output
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # Circular padding add the last sequence to the first
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Intialize the conv1d weight with kaiming he intialization method
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """Combined class of Embedder and Positional Encoding"""
    def __init__(self, c_in, d_model, dropout=0.0):
        """Constructor

        Args:
            c_in (_type_): input dimension of feature space within data 
            d_model (_type_): dimension that we would like the embedder to output
            dropout (float, optional): Dropout rate after embedding and positional encoding. Defaults to 0.0.
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # put everything together classic transformer techniques
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
