import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        #initialize the parent class
        super().__init__()
        #save d_model and vocab_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        #create the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        #scale the embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
    