from sklearn import svm
from rdkit import Chem
import numpy as np
import pandas as pd
import os

datadir = "/Users/yihyun/Code/"
prism_both = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test_molGNN_scdrug.csv"), index_col='smiles')
prism_train = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col='smiles')
prism_test_cl = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test_scdrug.csv"), index_col='smiles')

# convert smiles encoding to fingerprints
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
     tmp = prism_test_cl[cell_line].dropna()
     train_auc = np.array(tmp)

     ## train
     train_mol = tmp.index.tolist()
     train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
     train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
     train_fps = np.array(train_fps)

     X = train_fps
     y = train_auc
     model = svm.SVR()
     model.fit(X, y)

     pred_y = model.predict(test_fps)
     result_df.loc[:, cell_line] = pred_y

result_df.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_2.CaDRReS_CLsim_SVM/prediction/SVM_disjoint_pred.csv"), index = True)