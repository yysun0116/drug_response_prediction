### import module
import sys, timeit, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from MolecularGNN_smiles_train import MolecularGraphNeuralNetwork, Tester
import preprocess as pp

datadir = '/Users/yihyun/Code'
### set up parameters
if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')

task="regression" # target is a real value (e.g., energy eV).
dataset="PRISM"

radius=1
dim=50
layer_hidden=6
layer_output=6

batch_train=32
batch_test=32
lr=1e-4
lr_decay=0.99
decay_interval=10
iteration=1000

### import data
file_dir = os.path.join(datadir, "/drug_response_prediction/0.data/processed/")
data_train, data_test, N_fingerprints, fingerprints_dict = pp.create_datasets(file_dir, radius, device)

### load model (save on GPU, load on GPU)
torch.manual_seed(1234)
model = MolecularGraphNeuralNetwork(
            N_fingerprints, dim, layer_hidden, layer_output).to(device)
FILE = os.path.join(datadir, "drug_response_prediction/1.Computational_method_comparison/WL_GNN_model/save_model/molecularGNN_smiles_multi-output.pt")
#FILE = '/volume/yihyun/drug/baseline_model/save_model/molecularGNN_smiles_multi-output_norm.pt'
#FILE = '/volume/yihyun/drug/baseline_model/save_model/molecularGNN_smiles_multi-output_accLoss.pt'
#FILE = "/volume/yihyun/drug/baseline_model/save_model/molecularGNN_smiles_multi-output_excludeData.pt"
model.load_state_dict(torch.load(FILE))
model.to(device)


### predict 
tester = Tester(model)
torch.manual_seed(1234)
model.eval()
labels = []
outputs = []
N = len(data_test)
for i in range(0, N, batch_test):
    data_batch = list(zip(*data_test[i:i+batch_test]))
    predicted_values, correct_values, mask_i = model.forward_regressor(
                                        data_batch, train=False)
    if i ==0:
        labels = correct_values
        outputs = predicted_values
        mask = mask_i
    else:
        labels = torch.cat((labels, correct_values), 0)
        outputs= torch.cat((outputs, predicted_values), 0)
        mask = torch.cat((mask, mask_i), 0)

### save results
prism_test = pd.read_csv(file_dir + 'prism_drugBlind_test.csv', index_col = 'smiles')
pred_test_df = pd.DataFrame(np.matrix(outputs), index=prism_test.index, columns=prism_test.columns)
pred_test_df.to_csv(os.path.join(datadir, 'drug_response_prediction/1.Computational_method_comparison/WL_GNN_model/prediction/moleculargnn_smiles_pred.csv'), index=True)


### evaluation
from utils import masked_aCC, masked_mse
# cell-wise pcc
cl_wise_outputs = torch.transpose(outputs,0,1)
cl_wise_labels = torch.transpose(labels,0,1)
cl_wise_mask = torch.transpose(mask,0,1)
cl_wise_cc = masked_aCC(cl_wise_outputs, cl_wise_labels, cl_wise_mask)
masked_mse(cl_wise_outputs, cl_wise_labels, cl_wise_mask)
plt.boxplot(cl_wise_cc.to("cpu").data.numpy())
# percentile of pcc
print(np.percentile(cl_wise_cc, 25)) #Q1
print(np.percentile(cl_wise_cc, 50)) #Q2
print(np.percentile(cl_wise_cc, 75)) #Q3
print(np.percentile(cl_wise_cc, 25) - 1.5*(np.percentile(cl_wise_cc, 75)- np.percentile(cl_wise_cc, 25)))
print(np.percentile(cl_wise_cc, 75) + 1.5*(np.percentile(cl_wise_cc, 75)- np.percentile(cl_wise_cc, 25)))
print(max(cl_wise_cc), min(cl_wise_cc))


# drug-wise pcc
drug_wise_cc = masked_aCC(outputs, labels, mask)
masked_mse(outputs, labels, mask)
plt.boxplot(drug_wise_cc.to("cpu").data.numpy())
# percentile of pcc
print(np.percentile(drug_wise_cc, 25)) #Q1
print(np.percentile(drug_wise_cc, 50)) #Q2
print(np.percentile(drug_wise_cc, 75)) #Q3
print(np.percentile(drug_wise_cc, 25) - 1.5*(np.percentile(drug_wise_cc, 75)- np.percentile(drug_wise_cc, 25)))
print(np.percentile(drug_wise_cc, 75) + 1.5*(np.percentile(drug_wise_cc, 75)- np.percentile(drug_wise_cc, 25)))
print(max(drug_wise_cc), min(drug_wise_cc))

