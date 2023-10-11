# CaDRReS: similarity kernel of cell lines (import pre-computed kernel)
import pandas as pd
import numpy as np
import sys, os, pickle
np.set_printoptions(precision=2)
from collections import Counter
import importlib
from ipywidgets import widgets
import warnings
# model: 2.Combined_model_evaluation/2_1.CaDRReS_CLsim/CaDRReS_CLsim.py

with open ("/Users/yihyun/Code/drug_response_prediction/5.Discussion/5_2.dose_range/dose_range_mol.txt", "r") as f:
    dose_range_mol = f.read().split("\n")
all_kernel = pd.read_csv("/Users/yihyun/Code/drug_response_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/kernel_cl_train.csv", index_col = 0)
prism_train = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col = 'smiles')
train_mol = np.array(set(dose_range_mol)& set(prism_train.index))
prism_train = prism_train.loc[train_mol]
prism_test = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_test.csv", index_col = 'smiles')
test_mol = np.array(set(dose_range_mol)& set(prism_test.index))
prism_test = prism_test.loc[test_mol]
prism_test_cl = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_test_scdrug.csv", index_col = 'smiles')
prism_test_cl = prism_test_cl.loc[train_mol]

train_kernel = all_kernel.loc[prism_train.columns, prism_train.columns]
# kernel feature based only on training samples
cell_line_sample_list = prism_train.columns.tolist()
X_train = train_kernel.loc[cell_line_sample_list, cell_line_sample_list]
# observed drug response
Y_train = prism_train.T.loc[cell_line_sample_list]

# training CaDRReS (train_mol)
warnings.filterwarnings('ignore')
# specify output directry
output_dir = './result/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print ('Results will be saved in ', output_dir)

indication_weight_df = pd.DataFrame(np.ones(Y_train.shape), index=Y_train.index, columns=Y_train.columns)
obj_function = widgets.Dropdown(options=['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight'], description='Objetice function')
display(obj_function)
cadrres_model_dict, cadrres_output_dict = train_model(Y_train, X_train, Y_train, X_train, 
                                                      10, 0.0, 150000, 0.01, 
                                                      model_spec_name="cadrres-wo-sample-bias", 
                                                      save_interval=5000, output_dir=output_dir)
# save model
model_name = "same_dose_range"
#model_name = "same_dose_range_all"
print('Saving ' + output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLsim" + model_name))
pickle.dump(cadrres_model_dict, open(output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLsim" + model_name), 'wb'))
print('Saving ' + output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLsim" + model_name))
pickle.dump(cadrres_output_dict, open(output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLsim" + model_name), 'wb'))


