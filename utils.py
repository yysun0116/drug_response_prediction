import numpy as np
from rdkit import Chem

# evaluation functions
def masked_aCC(pred, label, mask):
    # multi-output average correlation coefficient with masking
    d_label = (label - np.nansum(label,1)/np.sum(mask, 1))
    d_pred = (pred - np.nansum(pred,1)/np.sum(mask, 1))
    x = np.nansum(np.multiply(d_label,d_pred), 1)
    y = np.sqrt(np.multiply(np.nansum(np.square(d_label), 1) , np.nansum(np.square(d_pred), 1)))
    #aCC = np.mean(x/y)
    return x/y

def masked_mse(pred, label, mask):
    # multi-output mse with masking
    return np.nansum(np.array(pred-label)**2, 1)/np.sum(np.array(mask),1)

def one_hot(vector, n):
    out = np.zeros(n)
    for i in vector:
        out[i] = 1
    return out

def RDKfp_convert(smiles_ls):
    mol_rdkit = list(map(Chem.MolFromSmiles,smiles_ls))
    fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in mol_rdkit]
    fps = np.array(fps)
    return fps