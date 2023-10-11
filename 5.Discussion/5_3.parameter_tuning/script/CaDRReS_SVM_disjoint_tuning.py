from sklearn import svm
from rdkit import Chem
import numpy as np
import pandas as pd
import os
from utils import masked_aCC, masked_mse
datadir = "/Users/yihyun/Code"

prism_both = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_disjoint_test.csv"), index_col='smiles')
prism_train = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_train.csv"), index_col='smiles')
prism_test_cl = pd.read_csv(os.path.join(datadir, "drug_response_prediction/0.data/processed/prism_cellBlind_test.csv"), index_col='smiles')
cadrres_pred = pd.read_csv(os.path.join(datadir, "drug_response_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/prediction/CaDRReS_CLsim_cellblind_pred.csv"), index_col='smiles')

## drug-wise
mask = np.matrix(np.isnan(prism_test_cl)==False, dtype=int)
label = np.matrix(prism_test_cl)
pred_cadrres = np.matrix(cadrres_pred)

### mse
cadrres_mse = masked_mse(pred_cadrres, label, mask)
cadrres_mse = cadrres_mse[np.isnan(cadrres_mse) == False]


# fingerprints
## train
prism_train_avail = prism_train.loc[[cl for cl in prism_test_cl.index if cl != prism_test_cl.index[98]]]
mol_train_out = prism_test_cl.index[98]
train_mol = prism_train_avail.index.tolist() + [mol_train_out]
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
train_fps = np.array(train_fps)
## test
test_mol = prism_both.index.tolist()
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
test_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in test_mol_rdkit]
test_fps = np.array(test_fps)
test_cl = prism_both.columns.tolist()

# CaDRReS-SVM with MSE Threshold (PRISM disjoint dataset)
mse_threshold = 0.0405
train_mol_tmp = prism_train_avail.index[np.where(cadrres_mse< mse_threshold)].to_list() + [mol_train_out]
cadrres_pred_train = cadrres_pred.loc[train_mol_tmp]
train_fps_tmp = train_fps[list(np.where(cadrres_mse< mse_threshold)[0]) + [98], :]


# evaluation result
tmp_result = np.empty((len(test_mol), len(test_cl)))
tmp_result[:] = np.nan
tmp_result_df = pd.DataFrame(tmp_result, columns=test_cl, index = test_mol)

for cell_line in test_cl:
    #print(cell_line)
    train_auc = np.array([auc if auc>=0 else 0 for auc in cadrres_pred_train[cell_line]])
    train_auc = np.array([auc if auc <=1 else 1 for auc in train_auc])

    X = train_fps_tmp
    y = train_auc
    model = svm.SVR()
    model.fit(X, y)

    pred_y = model.predict(test_fps)
    tmp_result_df.loc[:, cell_line] = pred_y

tmp_result_df.to_csv(os.path.join(datadir, "drug_response_prediction/5.Discussion/5_3.parameter_tuning/CaDRReS_SVM_disjoint_pred_tuning.csv"), index = True)