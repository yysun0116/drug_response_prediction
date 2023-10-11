# import modules
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit
from scipy.stats import pearsonr
import pickle
import os

datadir = '/Users/yihyun/Code'
# import training and testing data
print("import training and testing data...")
train_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
test_set = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_drugBlind_test.csv"), index_col="smiles")
cell_line_ls = train_set.columns.tolist()
print("training RandomForest model on %d cell lines" %(len(cell_line_ls)))

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


    # convert smiles encoding to fingerprints
    train_mol_rdkit = list(map(Chem.MolFromSmiles,train_mol))
    train_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in train_mol_rdkit]
    train_fps = np.array(train_fps)

    test_mol_rdkit = list(map(Chem.MolFromSmiles,test_mol))
    test_fps = [list(map(int, list(Chem.RDKFingerprint(x).ToBitString()))) for x in test_mol_rdkit]
    test_fps = np.array(test_fps)

    # svm model fitting
    X = train_fps
    y = train_auc
    model = RandomForestRegressor(random_state=1234)
    model.fit(X, y)

    # prediction
    pred_y = model.predict(test_fps)
    ## MSE
    result_df.loc[cell_line]["MSE"] = np.sum((test_auc - pred_y)**2)/len(test_auc)
    ## PCC
    result_df.loc[cell_line]["PCC"], _ = pearsonr(test_auc, pred_y)

    # save model
    filename = os.path.join(datadir, 'drug_sensitivity_prediction/1.Computational_method_comparison/ML_model/save_model/RF_RDKfp_'+cell_line+'.sav')
    pickle.dump(model, open(filename, 'wb'))

result_df.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/1.Computational_method_comparison/cell_line_eval/RF_RDKfp_result.csv"), index=True)
print("Done!")
