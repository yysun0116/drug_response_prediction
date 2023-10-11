from sklearn import svm
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import masked_mse, masked_aCC

## import prediction and ground truth
prism_train = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col='smiles')
prism_both = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_disjoint_test.csv", index_col='smiles')
prism_test_cl = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_cellBlind_test.csv", index_col='smiles')
cadrres_pred = pd.read_csv("/volume/yihyun/drug/MF_model/cadrres_CLsim_testCL_pred.csv", index_col='smiles')

## drug-wise
mask = np.matrix(np.isnan(prism_test_cl)==False, dtype=int)
label = np.matrix(prism_test_cl)
pred_cadrres = np.matrix(cadrres_pred)

## evaluation

### mse
cadrres_mse = masked_mse(pred_cadrres, label, mask)
cl_percentage = np.sum(np.array(np.isnan(prism_train) == False, dtype=int),1)/452
cl_percentage = cl_percentage[np.isnan(cadrres_mse) == False]
cadrres_mse = cadrres_mse[np.isnan(cadrres_mse) == False]

#prism_test_cl.loc[prism_test_cl.index[98]]
prism_train_avail = prism_train.loc[[cl for cl in prism_test_cl.index if cl != prism_test_cl.index[98]]]


# train SVM model with molecules under the MSE threshold
label = np.matrix(prism_both)
mask = np.matrix(np.isnan(prism_both)==False, dtype=int)


mol_train_out = prism_test_cl.index[98]
## train
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

# convert smiles encoding to fingerprints

mse_range = np.array(list(np.arange(2.0, 70.0, 0.1)/1000))


result = np.empty((len(mse_range), 6))
result[:] = np.nan
result_df = pd.DataFrame(result, columns=["MSE threshold", "number of training molecules", 
                                          "cell-wise MSE", "cell-wise aCC",
                                          "drug-wise MSE", "drug-wise aCC"])

for idx in range(len(mse_range)):
     print("MSE threshold = %.5f" %(mse_range[idx]))
     train_mol_tmp = prism_train_avail.index[np.where(cadrres_mse< mse_range[idx])].to_list() + [mol_train_out]
     cadrres_pred_train = cadrres_pred.loc[train_mol_tmp]
     train_fps_tmp = train_fps[list(np.where(cadrres_mse< mse_range[idx])[0]) + [98], :]


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
     
     pred = np.matrix(tmp_result_df)
     
     ## evaluation
     ### cell-wise
     cl_pcc = masked_aCC(pred.T, label.T, mask.T)
     cl_pcc = [p for p in cl_pcc if np.isnan(p) == False]
     cl_mse = masked_mse(pred.T, label.T, mask.T)
     cl_mse = [e for e in cl_mse if np.isnan(e) == False]

     ### drug-wise
     drug_pcc = masked_aCC(pred, label, mask)
     drug_pcc = [p for p in drug_pcc if np.isnan(p) == False]
     drug_mse = masked_mse(pred, label, mask)
     drug_mse = [e for e in drug_mse if np.isnan(e) == False]

     result_df.loc[idx, "MSE threshold"] = mse_range[idx]
     result_df.loc[idx, "number of training molecules"] = len(train_mol_tmp)
     result_df.loc[idx, "cell-wise MSE"] = np.mean(cl_mse)
     result_df.loc[idx, "cell-wise aCC"] = np.mean(cl_pcc)
     result_df.loc[idx, "drug-wise MSE"] = np.mean(drug_mse)
     result_df.loc[idx, "drug-wise aCC"] = np.mean(drug_pcc)

result_df.to_csv("/volume/yihyun/drug/parameter_tuning/mse_threshold.csv", index = True)


# check result under specific condition
condition = (result_df["MSE threshold"]<0.045) & (result_df["MSE threshold"]>0.040)
result_df[condition]

# visualization
## scatter plot of different MSE thresholds and cell-wise aCC
plt.scatter(result_df["MSE threshold"], result_df["cell-wise aCC"])
## scatter plot of different MSE thresholds and cell-wise MSE
plt.scatter(result_df["MSE threshold"], result_df["cell-wise MSE"])
##  scatter plot of different MSE thresholds and number of training molecules
plt.scatter(result_df["MSE threshold"], result_df["number of training molecules"])