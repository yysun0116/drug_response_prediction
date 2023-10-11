import numpy as np
import torch

def one_hot(vector, n):
    out = np.zeros(n)
    for i in vector:
        out[i] = 1
    return out

def masked_mse(pred, label, mask):
    # multi-output mse with masking
    return torch.sum(((pred-label)**2)*mask)/torch.sum(mask)

def masked_aCC(pred, label, mask):
    # multi-output average correlation coefficient with masking
    d_label = (label - (torch.sum(label*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
    d_pred = (pred - (torch.sum(pred*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
    x = torch.sum(d_label*d_pred, 1)
    y = torch.sqrt(torch.sum(d_label**2, 1) * torch.sum(d_pred**2, 1))
    aCC = torch.mean(x/y)
    #aCC = torch.mean(x/y)
    return x/y