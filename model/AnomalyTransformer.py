import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    """Entire encoder layer
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """Constructor

        Args:
            attention (NN): Attention layer
            d_model (Int): Model Dimension
            d_ff (Int, optional): feed forward dimension default to 4* d_model
            dropout (float, optional): dropout_rate for residual connections. Defaults to 0.1.
            activation (str, optional): Activation Function of the feedfoward. Defaults to "relu".
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x) #Attention layer residual connections
        y = x = self.norm1(x) #Post layer Norm
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) #first layer feedforward implemented using conv1d
        y = self.dropout(self.conv2(y).transpose(-1, 1)) #second layer feedforward implement using conv1d

        return self.norm2(x + y), attn, mask, sigma #return the residual connection then do a post layer norm


class Encoder(nn.Module):
    """The entire encoder consisiting of multiple encoder layer
    """
    def __init__(self, attn_layers, norm_layer=None):
        """Constructor

        Args:
            attn_layers (list of NN layers): The list of encoder layers
            norm_layer (NN layer, optional): The kind of normal layer we want to use in paper it is used as torch.nn.LayerNorm(d_model)
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) #This should not be attn_layers instead it is encoder layer
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = [] #append series for back_propgation of each encoder layer
        prior_list = [] #append prior for back_propgation of each encoder layer
        sigma_list = [] #append sigma for back_propgation of each encoder layer
        for attn_layer in self.attn_layers: 
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    """The full transformer layer
    """
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        """Constructor

        Args:
            win_size (int): window size (token size)
            enc_in (int): input dimension
            c_out (int): output dimension
            d_model (int, optional): The dimension after embedding. Defaults to 512.
            n_heads (int, optional): head of multihead attention. Defaults to 8.
            e_layers (int, optional): number of encoder layer. Defaults to 3.
            d_ff (int, optional): feed forward broadcast dimension. Defaults to 512.
            dropout (float, optional): Dropout rate for residual connection. Defaults to 0.0.
            activation (str, optional): Activation function for encoder layer. Defaults to 'gelu'.
            output_attention (bool, optional): Whether want to output the whole attention not just the reconstruction. Defaults to True.
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # The full encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Final Series reconstruction output with a linear layer projection
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
