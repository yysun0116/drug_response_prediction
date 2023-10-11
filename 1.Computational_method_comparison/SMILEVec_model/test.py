### import module
import os
import numpy as np
import pandas as pd
import torch
import pickle
from SMILESVec_mlp import SMILESVec_mlp

datadir = '/Users/yihyun/Code'
### set up parameters
if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')



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
# import preprocessed vectors of smiles
file_dir = os.path.join(datadir, "/drug_response_prediction/1.Computational_method_comparison/SMILEVec_model/data/")
with open(file_dir+"smiles_test.vec", "rb") as f:
        test_vec = pickle.loads(f.read())
# import auc data
test_set = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_drugBlind_test.csv", index_col="smiles")
    
# define mask
mask_df_test = (test_set.isna() == False).astype(int)
# filling missing value
test_set = test_set.fillna(value = -9)
# combine data
dataset_test = []
for i in range(len(test_set)):
    dataset_test.append((torch.FloatTensor(test_vec[i]), 
    torch.LongTensor(mask_df_test.loc[mask_df_test.index[i]]), torch.FloatTensor(test_set.loc[test_set.index[i]])))

### load model (save on GPU, load on GPU)
model = SMILESVec_mlp(dim, layer_output).to(device)
FILE = os.path.join(datadir, "drug_response_prediction/1.Computational_method_comparison/SMILEVec_model/save_model/SMILESVec_multi-output-mlp.pt")
model.load_state_dict(torch.load(FILE))
model.to(device)


### predict 
torch.manual_seed(1234)
model.eval()
labels = []
outputs = []
N = len(dataset_test)
for i in range(0, N, batch_test):
    data_batch = list(zip(*dataset_test[i:i+batch_test]))
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
pred_test_df = pd.DataFrame(np.matrix(outputs), index=test_set.index, columns=test_set.columns)
pred_test_df.to_csv(os.path.join(datadir, 'drug_response_prediction/1.Computational_method_comparison/WL_GNN_model/prediction/SMILESVec_multi_out_pred.csv'), index=True)


