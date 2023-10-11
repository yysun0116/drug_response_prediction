# CaDRReS-SVM (external)
from sklearn import svm
from rdkit import Chem
import rdkit
import numpy as np
import pandas as pd

with open ("/volume/yihyun/drug/baseline_model/data/dose_range_mol.txt", "r") as f:
    dose_range_mol = f.read().split("\n")
prism_train = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col = 'smiles')
train_mol = np.array(set(dose_range_mol)& set(prism_train.index))
prism_train = prism_train.loc[train_mol]
GDSC2_external = pd.read_csv("/volume/yihyun/drug/GDSC/GDSC2_external_auc.csv", index_col = 0)
cadrres_pred = pd.read_csv("/volume/yihyun/drug/combined_model/result/cadrres_GDSC_sameDRmol_pred.csv", index_col='smiles')
ccle = pd.read_csv("/volume/yihyun/drug/CCLE_expression.csv", index_col = 0)
external_test_cl = list(set(ccle.index.tolist()) & set(GDSC2_external.columns.tolist()))
# convert smiles encoding to fingerprints
## train
train_mol = prism_train.index.tolist()
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
train_fps = np.array(train_fps)
## test
test_mol = GDSC2_external.index.tolist()
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
test_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in test_mol_rdkit]
test_fps = np.array(test_fps)
test_cl = cadrres_pred.columns.tolist()

# evaluation result
result = np.empty((len(test_mol), len(test_cl)))
result[:] = np.nan
result_df = pd.DataFrame(result, columns=test_cl, index = test_mol)

for cell_line in test_cl:
     print(cell_line)
     train_auc = np.array([auc if auc>=0 else 0 for auc in cadrres_pred[cell_line]])
     train_auc = np.array([auc if auc <=1 else 1 for auc in train_auc])

     X = train_fps
     y = train_auc
     model = svm.SVR()
     model.fit(X, y)

     pred_y = model.predict(test_fps)
     result_df.loc[:, cell_line] = pred_y

result_df[external_test_cl].to_csv("/volume/yihyun/drug/GDSC/cadrres_svm_test_external_sameDRmol.csv", index = True)