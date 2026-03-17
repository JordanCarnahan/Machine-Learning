#https://docs.pytorch.org/docs/stable/nn.init.html
import torch
import torch.nn as nn
import math
import embedding
w = torch.empty(3, 5)
nn.init.xavier_uniform_(w)
print(w, "\n")

z = torch.empty(3, 5)
nn.init.kaiming_uniform_(z, mode="fan_in", nonlinearity="relu")
print(z, "\n")

y = torch.empty(3, 5)
nn.init.kaiming_uniform_(y, a=0, nonlinearity="leaky_relu")
nn.init.kaiming_uniform_(embedding.weight, a=0, nonlinearity="leaky_relu")
print(y, "\n")    

a = torch.empty(3, 5, dtype=torch.float32)
print(a, "\n")

# Padding
# <pad> or [PAD]
# Used to fill sequences to a uniform length within a batch.
# The loss function typically ignores these tokens.

# Unknown
# <unk> or [UNK]
# Represents words encountered during inference or validation
# that were not in the model's original vocabulary.

# Begin-of-Sentence / Start-of-Text
# <bos>, <SOS>
# Indicates the start of a sentence or sequence.

# End-of-Sentence / End-of-Text
# <eot>, <EOS>
# Indicates the end of a sentence or sequence.

# Mask
# [MASK]
# Used in masked language modeling (e.g., BERT).
# Some tokens are hidden and the model must predict them.