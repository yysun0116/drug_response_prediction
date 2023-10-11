# svm RDKfp testing
from sklearn import svm
import numpy as np
import pandas as pd
from rdkit import DataStructs, Chem
from scipy.stats import pearsonr
import pickle
import os

datadir = '/Users/yihyun/Code'
train_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
test_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_drugBlind_test.csv"), index_col="smiles")
cell_line_ls = train_set.columns.tolist()
cell_line_ls = train_set.columns.tolist()
train_mol_all = train_set.index.tolist()
test_mol_all =  test_set.index.tolist()

# calculation of drug similarity
## train
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol_all))
train_fps = [Chem.RDKFingerprint(x) for x in train_mol_rdkit]
## test
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol_all))
test_fps = [Chem.RDKFingerprint(x) for x in test_mol_rdkit]
mol_sim_mat = np.zeros((len(test_mol_all), len(train_mol_all)))
for i in range(len(test_mol_all)):
    for j in range(len(train_mol_all)):
        sim_score = DataStructs.FingerprintSimilarity(test_fps[i], train_fps[j])
        mol_sim_mat[i, j] = sim_score
test_mol_sim_df = pd.DataFrame(mol_sim_mat, index=test_mol_all, columns=train_mol_all)

n = 0
for cell_line in cell_line_ls:
    n += 1
    # load model
    filename = os.path.join(datadir,"drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/save_model/svm_simMat_"+cell_line+".sav")
    model = pickle.load(open(filename, 'rb'))

    # fingerprints
    train_mol = train_set[cell_line].dropna().index.tolist()
    test_mol = test_set[cell_line].dropna().index.tolist()
    # similarity matrix
    test_sim = np.array(test_mol_sim_df.loc[test_mol][train_mol])

    # save data
    pred_test_mat = np.empty((len(test_fps), 3),)
    pred_test_mat[:] = np.nan
    pred_test_df = pd.DataFrame(pred_test_mat, columns=["cell_line", "drug", "pred_auc"])

    # prediction
    pred_test_df["pred_auc"] = model.predict(test_sim)
    pred_test_df["drug"] = test_mol
    pred_test_df["cell_line"] = cell_line

    if n ==1:
        all_pred_auc = pred_test_df
    else:
        all_pred_auc = pd.concat([all_pred_auc, pred_test_df])

all_pred_auc_wide = all_pred_auc.pivot(index='drug', columns='cell_line', values='pred_auc')
all_pred_auc_wide.to_csv(os.path.join(datadir,
                                      "drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/prediction/svm_simMat_DrugBlind_pred.csv"))
