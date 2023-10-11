# import modules
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from rdkit import DataStructs, Chem
import rdkit
from scipy.stats import pearsonr
import pickle
import time
import os


datadir = '/Users/yihyun/Code'
# import training and testing data
print("import training and testing data...")
train_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
test_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_drugBlind_test.csv"), index_col="smiles")
cell_line_ls = train_set.columns.tolist()
train_mol_all = train_set.index.tolist()
test_mol_all =  test_set.index.tolist()

# calculation of drug similarity
## train
train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol_all))
train_fps = [Chem.RDKFingerprint(x) for x in train_mol_rdkit]

start = time.time()
mol_sim_mat = np.zeros((len(train_mol_all), len(train_mol_all)))
for i in range(len(train_mol_all)):
    if (i+1)%100 == 0:
        print ("{} of {} ({:.2f})s".format(i+1, len(train_mol_all), time.time()-start))
        start = time.time()
    for j in range(len(train_mol_all)):
        sim_score = DataStructs.FingerprintSimilarity(train_fps[i], train_fps[j])
        mol_sim_mat[i, j] = sim_score
train_mol_sim_df = pd.DataFrame(mol_sim_mat, index=train_mol_all, columns=train_mol_all)

## test
test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol_all))
test_fps = [Chem.RDKFingerprint(x) for x in test_mol_rdkit]
start = time.time()
mol_sim_mat = np.zeros((len(test_mol_all), len(train_mol_all)))
for i in range(len(test_mol_all)):
    if (i+1)%100 == 0:
        print ("{} of {} ({:.2f})s".format(i+1, len(test_mol_all), time.time()-start))
        start = time.time()
    for j in range(len(train_mol_all)):
        sim_score = DataStructs.FingerprintSimilarity(test_fps[i], train_fps[j])
        mol_sim_mat[i, j] = sim_score
test_mol_sim_df = pd.DataFrame(mol_sim_mat, index=test_mol_all, columns=train_mol_all)

# evaluation result
result = np.empty((len(cell_line_ls), 2))
result[:] = np.nan
result_df = pd.DataFrame(result, columns=["MSE", "PCC"], index = cell_line_ls)


for cell_line in cell_line_ls:
    print("training on %s cell line" %(cell_line))
    train_auc = train_set[cell_line].dropna().tolist()
    train_mol = train_set[cell_line].dropna().index.tolist()
    test_auc = test_set[cell_line].dropna().tolist()
    test_mol = test_set[cell_line].dropna().index.tolist()

    # similarity of molecules in the training cell line
    train_sim = np.array(train_mol_sim_df.loc[train_mol][train_mol])
    test_sim = np.array(test_mol_sim_df.loc[test_mol][train_mol])


    # svm model fitting
    X = train_sim
    y = train_auc
    model = RandomForestRegressor()
    model.fit(X, y)

    # prediction
    pred_y = model.predict(test_sim)
    ## MSE
    result_df.loc[cell_line]["MSE"] = np.sum((test_auc - pred_y)**2)/len(test_auc)
    ## PCC
    result_df.loc[cell_line]["PCC"], _ = pearsonr(test_auc, pred_y)

    # save model
    filename = os.path.join(datadir, 'drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/save_model/RF_simMat_'+cell_line+'.sav')
    pickle.dump(model, open(filename, 'wb'))

result_df.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/cell_line_eval/RF_simMat_result.csv"), index=True)
print("Done!")
