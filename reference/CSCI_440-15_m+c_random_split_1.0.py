import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

# 1. Define a sample dataset (replace with your actual dataset)
full_dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))

# 2. Define the desired lengths for the splits
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
lengths = [train_size, test_size]

# 3. Perform the random split
# For reproducibility, you can use a generator:
# generator = torch.Generator().manual_seed(42)
# train_dataset, test_dataset = random_split(full_dataset, lengths, generator=generator)
train_dataset, test_dataset = random_split(full_dataset, lengths)

# 4. Use the subsets with DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Original dataset length: {len(full_dataset)}")
print(f"Training dataset length: {len(train_dataset)}")
print(f"Testing dataset length: {len(test_dataset)}")
