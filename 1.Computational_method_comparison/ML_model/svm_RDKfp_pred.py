# svm RDKfp testing
from sklearn import svm
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import pearsonr
import pickle
import os

datadir = '/Users/yihyun/Code'
train_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
test_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test.csv"), index_col="smiles")
cell_line_ls = train_set.columns.tolist()

n = 0
for cell_line in cell_line_ls:
    n += 1
    # load model
    filename = os.path.join(datadir,"drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/save_model/svm_RDKfp_"+cell_line+".sav")
    model = pickle.load(open(filename, 'rb'))

    # fingerprints
    test_mol = test_set[cell_line].dropna().index.tolist()
    test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
    test_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in test_mol_rdkit]
    test_fps = np.array(test_fps)

    # save data
    pred_test_mat = np.empty((len(test_fps), 3),)
    pred_test_mat[:] = np.nan
    pred_test_df = pd.DataFrame(pred_test_mat, columns=["cell_line", "drug", "pred_auc"])

    # prediction
    pred_test_df["pred_auc"] = model.predict(test_fps)
    pred_test_df["drug"] = test_mol
    pred_test_df["cell_line"] = cell_line

    if n ==1:
        all_pred_auc = pred_test_df
    else:
        all_pred_auc = pd.concat([all_pred_auc, pred_test_df])

all_pred_auc_wide = all_pred_auc.pivot(index='drug', columns='cell_line', values='pred_auc')
all_pred_auc_wide.to_csv(os.path.join(datadir,
                                      "drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/prediction/svm_RDKfp_DrugBlind_pred.csv"))
