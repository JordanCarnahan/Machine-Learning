# %%
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        #initialize the parent class
        super().__init__()
        #save d_model, seq_len, and dropout
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create positional encodings
        pos_enc = torch.zeros(seq_len, d_model)
        #shape: (seq_len, d_model)
        #from 1D matrix (seq_len,) to 2D matrix (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #Creates the scale factors that set each even dimension’s sine/cosine wavelength.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #formula 1: PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        #formula 2: PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, seq_len, d_model)
        pe = pos_enc.unsqueeze(0)
        #register pos_enc as a buffer that is not a parameter
        self.register_buffer('pos_enc', pos_enc)

        def forward(self, x):
            x = x + self.pos_enc[:, :x.size(1), :]
            return self.dropout(x)