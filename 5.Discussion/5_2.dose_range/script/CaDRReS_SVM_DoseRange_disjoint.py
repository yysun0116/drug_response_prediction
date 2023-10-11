# training CaDRReS-SVM
from sklearn import svm
from rdkit import Chem
import rdkit
import numpy as np
import pandas as pd
prism_train = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col = 'smiles')
train_mol = np.array(set(dose_range_mol)& set(prism_train.index))
prism_train = prism_train.loc[train_mol]
prism_test = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_test.csv", index_col = 'smiles')
test_mol = np.array(set(dose_range_mol)& set(prism_test.index))
prism_test = prism_test.loc[test_mol]
prism_test_cl = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_test_scdrug.csv", index_col = 'smiles')
prism_test_cl = prism_test_cl.loc[train_mol]
prism_both = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_disjoint_test.csv", index_col='smiles')
prism_both = prism_both.loc[test_mol]
cadrres_pred = pd.read_csv("/volume/yihyun/drug/combined_model/result/cadrres_CLsim_testCL_sameDRmol_pred.csv", index_col='smiles')

# convert smiles encoding to fingerprints
## train
train_mol = prism_train.index.tolist()
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
train_fps = np.array(train_fps)
## test
test_mol = prism_both.index.tolist()
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
test_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in test_mol_rdkit]
test_fps = np.array(test_fps)
test_cl = prism_both.columns.tolist()

# evaluation result
result = np.empty((len(test_mol), len(test_cl)))
result[:] = np.nan
result_df = pd.DataFrame(result, columns=test_cl, index = test_mol)

for cell_line in test_cl:
     print(cell_line)
     train_auc = np.array([auc if auc>=0 else 0 for auc in cadrres_pred[cell_line]])
     train_auc = np.array([auc if auc <=1 else 1 for auc in train_auc])
     train_auc

     X = train_fps
     y = train_auc
     model = svm.SVR()
     model.fit(X, y)

     pred_y = model.predict(test_fps)
     result_df.loc[:, cell_line] = pred_y
result_df.to_csv("/Users/yihyun/Code/drug_response_prediction/5.Discussion/5_2.dose_range/prediction/cadrres_svm_test_both_sameDRmol.csv", index = True)