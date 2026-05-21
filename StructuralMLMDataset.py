import torch
import numpy as np
from torch.utils.data import Dataset

class StructuralMLMDataset(Dataset):
    def __init__(self, matrix, mask_token_id=0, mask_prob=0.15):
        self.matrix = matrix
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        src = np.copy(self.matrix[idx])
        tgt = np.copy(self.matrix[idx])
        
        # Randomly replace 15% of features with [MASK] (0) to create the prediction quiz
        for i in range(len(src)):
            if np.random.rand() < self.mask_prob:
                src[i] = self.mask_token_id  
                
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)