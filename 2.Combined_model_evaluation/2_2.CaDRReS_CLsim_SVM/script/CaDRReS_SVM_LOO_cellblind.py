from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from rdkit import Chem
import numpy as np
import os
import pandas as pd

# import data
datadir = "/Users/yihyun/Code/"
prism_both = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_disjoint_test.csv"), index_col='smiles')
prism_train = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col='smiles')
pred_df = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/prediction/CaDRReS_CLsim_cellblind_pred.csv"), index_col='smiles')
test_cl = prism_both.columns.to_list()

loo = LeaveOneOut()
# convert smiles encoding to fingerprints
train_mol = prism_train.index.tolist()
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
train_fps = np.array(train_fps)

# evaluation result
result = np.empty((len(train_mol), len(test_cl)))
result[:] = np.nan
result_df = pd.DataFrame(result, columns=test_cl, index = train_mol)

for cell_line in test_cl:
     #cell_line = test_cl[0]
     cadrres_pred = pred_df.loc[test_cl].T
     
     train_auc = np.array([auc if auc>=0 else 0 for auc in cadrres_pred[cell_line]])

     for i, (train_index, test_index) in enumerate(loo.split(train_fps)):
          print(f"Fold {i}:")
          # svm model fitting
          X = train_fps[train_index,:]
          y = train_auc[train_index]
          model = svm.SVR()
          model.fit(X, y)

          test_fps = train_fps[test_index,:]
          pred_y = model.predict(test_fps)
          result_df.loc[np.array(train_mol)[test_index], cell_line] = pred_y

result_df.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_2.CaDRReS_CLsim_SVM/prediction/CaDRReS_SVM_LOO_cellblind_pred.csv"), index = True)

