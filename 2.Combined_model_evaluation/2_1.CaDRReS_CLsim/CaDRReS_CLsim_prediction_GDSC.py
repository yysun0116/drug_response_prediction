#testing set kernel
from rdkit import DataStructs, Chem
import time
import numpy as np
from scipy import stats
import pandas as pd
from CaDRReS_CLsim import load_model, predict_from_model
import os


datadir = '/Users/yihyun/Code'
prism_train = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col="smiles")
GDSC2_external = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/GDSC2_external_auc.csv"), index_col = 0)
all_kernel = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/kernel_cl_train.csv"), index_col = 0)
ccle = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/raw/CCLE_expression.csv"), index_col = 0)

train_cl = prism_train.columns.tolist()
external_test_cl = list(set(ccle.index.tolist()) & set(GDSC2_external.columns.tolist()))
test_kernel = all_kernel.loc[train_cl+external_test_cl, train_cl+external_test_cl]

# predict
model_spec_name = "cadrres-wo-sample-bias_CLSim"
model_dir = os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/save_model/")
model_file = model_dir + '{}_param_dict.pickle'.format(model_spec_name)
cadrres_model = load_model(model_file)
print('Predicting drug response using CaDRReS: {}'.format(model_spec_name))
pred_df, P_test_df= predict_from_model(cadrres_model, test_kernel, "cadrres-wo-sample-bias")
print('done!')

# save prediction
pred_df.loc[external_test_cl].T.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/prediction/CaDRReS_CLsim_external_cl_pred.csv"), index = True)

