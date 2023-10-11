import pandas as pd
import numpy as np
# model: 2.Combined_model_evaluation/2_1.CaDRReS_CLsim/CaDRReS_CLsim.py

# external testing (GDSC2)
GDSC2_external = pd.read_csv("/volume/yihyun/drug/GDSC/GDSC2_external_auc.csv", index_col = 0)
ccle = pd.read_csv("/volume/yihyun/drug/CCLE_expression.csv", index_col = 0)
all_kernel = pd.read_csv("/volume/yihyun/drug/MF_model/kernel_cl_train.csv", index_col = 0)
train_cl = prism_train.columns.tolist()
external_test_cl = list(set(ccle.index.tolist()) & set(GDSC2_external.columns.tolist()))
test_kernel = all_kernel.loc[train_cl+GDSC2_external.columns.tolist(), train_cl+GDSC2_external.columns.tolist()]
# loading model
model_name = "same_dose_range"
model_spec_name = "cadrres-wo-sample-bias_CLsim"

model_dir = "/volume/yihyun/drug/combined_model/result/"
model_file = model_dir + '{}_param_dict.pickle'.format(model_spec_name + model_name)
cadrres_model = load_model(model_file)
print('Predicting drug response using CaDRReS: {}'.format(model_spec_name))

pred_df, P_test_df= predict_from_model(cadrres_model, test_kernel, "cadrres-wo-sample-bias")
pred_df.T[external_test_cl].to_csv("/volume/yihyun/drug/combined_model/result/cadrres_GDSC_sameDRmol_pred.csv", index = True)
print('done!')
