import torch
import torch.nn.functional as F
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, embed_size, d_model, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # Dimensionality of input embeddings
        self.d_model = d_model  # Dimensionality of the model (hidden layers)
        self.heads = heads
        self.head_dim = d_model // heads  # Head dimension

        # Ensure the d_model is divisible by the number of heads
        assert self.head_dim * heads == d_model, "d_model must be divisible by number of heads"

        # Linear layers for queries, keys, and values (projecting from embed_size to d_model)
        self.values = nn.Linear(embed_size, d_model)  # Embedding to model size (W_V)
        self.keys = nn.Linear(embed_size, d_model)    # Embedding to model size (W_K)
        self.queries = nn.Linear(embed_size, d_model) # Embedding to model size (W_Q)

        # Output linear layer to combine multi-head output
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]  # Batch size
        seq_len = x.shape[1]     # Sequence length

        # Step 1: Pass the input word embeddings (x) through linear layers
        queries = self.queries(x)  # (batch_size, seq_len, d_model)
        keys = self.keys(x)        # (batch_size, seq_len, d_model)
        values = self.values(x)    # (batch_size, seq_len, d_model)

        # Step 2: Split the embeddings into multiple heads
        queries = queries.reshape(batch_size, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.heads, self.head_dim)

        # Transpose to get the shape (batch_size, heads, seq_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Step 3: Calculate attention scores (Q * K^T)
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # Shape: (batch_size, heads, seq_len, seq_len)

        # Step 4: Apply mask (if any)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # Apply mask

        # Step 5: Normalize the energy scores to get attention weights
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)

        # Step 6: Compute the weighted sum of values
        out = torch.matmul(attention, values)  # (batch_size, heads, seq_len, head_dim)

        # Step 7: Reshape the output back to (batch_size, seq_len, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, self.heads * self.head_dim)

        # Step 8: Apply the final linear layer
        out = self.fc_out(out)

        return out

# Test the SelfAttention module
if __name__ == "__main__":
    embed_size = 128  # Dimensionality of input embeddings (e.g., word embeddings)
    d_model = 256     # Dimensionality of the model (hidden layers)
    heads = 8         # Number of attention heads
    seq_length = 10   # Length of the sequence (e.g., sentence length)
    batch_size = 4    # Number of samples in the batch

    # Random input tensor representing a batch of word embeddings
    x = torch.rand((batch_size, seq_length, embed_size))  # (batch_size, seq_len, embed_size)

    # Instantiate the SelfAttention module
    self_attention = SelfAttention(embed_size, d_model, heads)

    # Perform forward pass
    out = self_attention(x)

    print("Output shape:", out.shape)  # Expected shape: (batch_size, seq_len, d_model)