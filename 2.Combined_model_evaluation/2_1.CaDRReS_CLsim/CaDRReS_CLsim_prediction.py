#testing set kernel
from rdkit import DataStructs, Chem
import time
import numpy as np
from scipy import stats
import pandas as pd
from CaDRReS_CLsim import load_model, predict_from_model
import os


datadir = '/Users/yihyun/Code'
prism_test = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test.csv"), index_col="smiles")
prism_train = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
prism_test_cl = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test_scdrug.csv"), index_col = 'smiles')
prism_test_both = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test_molGNN_scdrug.csv"), index_col='smiles')
all_kernel = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/kernel_cl_train.csv"), index_col = 0)


train_cl = prism_train.columns.tolist()
test_cl = prism_test_cl.columns.tolist()
test_kernel = all_kernel.loc[train_cl+test_cl, train_cl+test_cl]

# predict
# model_spec_name = "cadrres-wo-sample-bias_CLSim" # cell-blind set prediction
model_spec_name = "cadrres-wo-sample-bias_CLSim_allMol" # cell-blind set + disjoint set prediction
model_dir = os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/save_model/")
model_file = model_dir + '{}_param_dict.pickle'.format(model_spec_name)
cadrres_model = load_model(model_file)
print('Predicting drug response using CaDRReS: {}'.format(model_spec_name))
pred_df, P_test_df= predict_from_model(cadrres_model, test_kernel, "cadrres-wo-sample-bias")
print('done!')

# save prediction
## cell-blind set prediction
# cadrres_testCL = pred_df.loc[test_cl].T
# cadrres_testCL.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/prediction/CaDRReS_CLsim_cellblind_pred.csv"), index = True)

## disjoint set prediction
cadrres_both_pred = pred_df.loc[prism_test_both.columns.tolist(), prism_test_both.index.tolist()].T
cadrres_both_pred.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/prediction/CaDRReS_CLsim_disjoint_pred.csv"), index = True)
