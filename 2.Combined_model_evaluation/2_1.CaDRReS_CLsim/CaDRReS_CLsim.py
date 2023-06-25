# kernel calculation
## from CaDRRes-sc
import pandas as pd
import numpy as np
from scipy import stats
import time

def log2_exp(exp_df):
    """Calculate log2 gene expression
    """

    return np.log2(exp_df + 1)

# TODO: add pseudo count for RNA-seq data
def normalize_log2_mean_fc(log2_exp_df):
    """Calculate gene expression fold-change based on median of each genes. The sample size should be large enough (>10).
    """

    return (log2_exp_df.T - log2_exp_df.mean(axis=1)).T, pd.DataFrame(log2_exp_df.mean(axis=1), columns=['median'])

def normalize_log2_mean_fc_with_ref(log2_exp_df, log2_ref_exp_df):
    """Calculate gene expression fold-change based on median of each genes. 
    This should not be used if the data come from different experiments.
    """

    common_genes = set(log2_ref_exp_df.index).intersection(log2_exp_df.index)
    log2_exp_df = log2_exp_df.loc[common_genes]
    log2_ref_exp_df = log2_ref_exp_df.loc[common_genes]

    return (log2_exp_df.T - log2_ref_exp_df.mean(axis=1)).T, pd.DataFrame(log2_ref_exp_df.mean(axis=1), columns=['median'])

def normalize_L1000_suite():
    """
    """

# TODO: make this run in parallel
def calculate_kernel_feature(log2_median_fc_exp_df, ref_log2_median_fc_exp_df, gene_list):
    common_genes = [g for g in gene_list if (g in log2_median_fc_exp_df.index) and (g in ref_log2_median_fc_exp_df.index)]
    
    print ('Calculating kernel features based on', len(common_genes), 'common genes')

    print (log2_median_fc_exp_df.shape, ref_log2_median_fc_exp_df.shape)
    
    sample_list = list(log2_median_fc_exp_df.columns)
    ref_sample_list = list(ref_log2_median_fc_exp_df.columns)

    exp_mat = np.array(log2_median_fc_exp_df.loc[common_genes], dtype='float')
    ref_exp_mat = np.array(ref_log2_median_fc_exp_df.loc[common_genes], dtype='float')

    sim_mat = np.zeros((len(sample_list), len(ref_sample_list)))

    start = time.time()
    for i in range(len(sample_list)):
        if (i+1)%100 == 0:
            print ("{} of {} ({:.2f})s".format(i+1, len(sample_list), time.time()-start))
            start = time.time()
        for j in range(len(ref_sample_list)):
            p_cor, _ = stats.pearsonr(exp_mat[:,i], ref_exp_mat[:,j])
            sim_mat[i, j] = p_cor

    return pd.DataFrame(sim_mat, columns=ref_sample_list, index=sample_list)

# model
## drug/cell line similarity 
## from CaDRReS-Sc/model.py
import pandas as pd
import numpy as np
np.set_printoptions(precision=2)
from collections import Counter
from ipywidgets import widgets
import warnings
warnings.filterwarnings('ignore')
import sys, os, pickle, time

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.python.framework import ops
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def load_model(model_fname):

    """Load a pre-trained model
	:param model_fname: File name of the model
	:return: model_dict contains model information
    """

    model_dict = pickle.load(open(model_fname, 'br'))

    return model_dict


# def get_model_param(pg_space):

    """Get model paramters
    """

# def get_training_info(pg_space):

    """Get training information
    """

def predict_from_model(model_dict, test_kernel_df, model_spec_name='cadrres-wo-sample-bias'):

    """Make a prediction of testing samples. Only for the model without sample bias.
    """

    # TODO: add other model types and update the scrip accordingly
    if model_spec_name not in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        return None

    sample_list = list(test_kernel_df.index)
    
    # Read drug list from model_dict
    drug_list = model_dict['drug_list']
    kernel_sample_list = model_dict['kernel_sample_list']

    # Prepare input
    X = np.matrix(test_kernel_df[kernel_sample_list])

    # Make a prediction
    b_q = model_dict['b_Q']
    WP = model_dict['W_P']
    WQ = model_dict['W_Q']

    n_dim = WP.shape[1]

    pred = b_q.T + (X * WP) * WQ.T
    pred = (pred+80)/240 # convert sensitivity score to IC50
    pred_df = pd.DataFrame(pred, sample_list, drug_list)

    # Projections
    P_test = X * WP
    P_test_df = pd.DataFrame(P_test, index=sample_list, columns=range(1,n_dim+1))  
    
    return pred_df, P_test_df

def calculate_baseline_prediction(obs_resp_df, train_sample_list, drug_list, test_sample_list):

    """Calculate baseline prediction, i.e., for each drug, predict the average response.
    """

    repeated_val = np.repeat([obs_resp_df.loc[train_sample_list, drug_list].mean().values], len(test_sample_list), axis=0)
    return pd.DataFrame(repeated_val, index=test_sample_list, columns=drug_list)


##########################
##### Model training #####
##########################

##### Utility functions #####

def create_placeholders(n_x_features, n_y_features, sample_weight=False):

    """
    Create placeholders for model inputs
    """

    # gene expression
    X = tf.placeholder(tf.float32, [None, n_x_features])
    # drug response
    Y = tf.placeholder(tf.float32, [None, n_y_features])
    if sample_weight:
        # for logistic weight based on maximum drug dosage
        O = tf.placeholder(tf.float32, [None, None])
        # for indication-specific weight
        D = tf.placeholder(tf.float32, [None, None])
        return X, Y, O, D
    else:
        return X, Y

def initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dimensions, seed):

    """
    Initialize parameters
    Depending on the objective function, b_P might not be used in the later step.
    """

    parameters = {}

    parameters['W_P'] = tf.Variable(tf.truncated_normal([n_x_features, n_dimensions], stddev=0.2, mean=0, seed=seed), name="W_P")
    parameters['W_Q'] = tf.Variable(tf.truncated_normal([n_y_features, n_dimensions], stddev=0.2, mean=0, seed=seed), name="W_Q")
    parameters['b_P'] = tf.get_variable('b_P', [n_samples, 1], initializer = tf.zeros_initializer())
    parameters['b_Q'] = tf.get_variable('b_Q', [n_drugs, 1], initializer = tf.zeros_initializer())

    return parameters

def inward_propagation(X, Y, parameters, n_samples, n_drugs, model_spec_name):

    """
    Define base objective function
    """

    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)

    b_P_mat = tf.matmul(parameters['b_P'], tf.convert_to_tensor(np.ones(n_drugs).reshape(1, n_drugs), np.float32))
    b_Q_mat = tf.transpose(tf.matmul(parameters['b_Q'], tf.convert_to_tensor(np.ones(n_samples).reshape(1, n_samples), np.float32)))

    if model_spec_name == 'cadrres':
        S = tf.add(b_Q_mat, tf.add(b_P_mat, tf.matmul(P, tf.transpose(Q))))
    elif model_spec_name in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
        S = tf.add(b_Q_mat, tf.matmul(P, tf.transpose(Q)))
    # TODO: add the model without both drug and sample biases
    # elif model_spec_name == 'cadrres-wo-bias':
    #     S = tf.matmul(P, tf.transpose(Q))
    else:
        S = None

    return S

def get_latent_vectors(X, Y, parameters):

    """
    Get latent vectors of cell line (P) and drug (Q) on the pharmacogenomic space
    """

    W_P = parameters['W_P']
    W_Q = parameters['W_Q']
    P = tf.matmul(X, W_P)
    Q = tf.matmul(Y, W_Q)
    return P, Q

##### Predicting function #####

def predict(X, Y, S_obs, parameters_trained, X_sample_list, model_spec_name, is_train):

    """
    Make a prediction and calculate cost. This function is used in the training step.
    """

    n_samples = len(X_sample_list)
    n_drugs = Y.shape[1]
    
    W_P = parameters_trained['W_P']
    W_Q = parameters_trained['W_Q']
    P = np.matmul(X, W_P)
    Q = np.matmul(Y, W_Q)

    if model_spec_name in ['cadrres-wo-sample-bias', 'cadrres-wo-sample-bias-weight']:
    
        b_Q = parameters_trained['b_Q']

        b_Q_mat = np.transpose(np.matmul(b_Q, np.ones(n_samples).reshape(1, n_samples)))
        S = b_Q_mat + np.matmul(P, np.transpose(Q))

        cost = np.nanmean(np.square(S - S_obs))/2.0

    elif model_spec_name == 'cadrres':
 
        b_Q = parameters_trained['b_Q']
        b_P = parameters_trained['b_P']

        if is_train:

            b_P_est = b_P

        else:
            # estimate sample bias
            b_P_est = np.matmul(X, b_P) 
            # copy bias for seen samples
            for u, s_name in enumerate(X_sample_list):
                if s_name in parameters_trained['sample_list_train']:
                    s_idx = parameters_trained['sample_list_train'].tolist().index(s_name)
                    b_P_est[u, 0] = b_P[s_idx, 0]

        b_P_mat = np.matmul(b_P_est, np.ones(n_drugs).reshape(1, n_drugs))
        b_Q_mat = np.transpose(np.matmul(b_Q, np.ones(n_samples).reshape(1, n_samples)))
        S = b_Q_mat + b_P_mat + np.matmul(P, np.transpose(Q))        

        cost = np.nanmean(np.square(S - S_obs))/2.0

    return S, cost

##### Training function #####

def train_model(train_resp_df, train_feature_df, test_resp_df, test_feature_df, n_dim, lda, max_iter, l_rate, model_spec_name='cadrres-wo-sample-bias', seed=1, save_interval=1000, output_dir='output'):

    """
    Train a model. This is for the original cadrres and cadrres-wo-sample-bias
    :param train_resp_df: drug response training data
    :param train_feature_df: kernel feature training data
    :param test_resp_df: drug response testing data
    :param test_feature_df: kernel feature testing data
    :param n_dim: number of dimension of the latent space
    :param lda: regularization factor
    :param max_iter: maximum iteration
    :param l_rate: learning rate
    :param model_spec_name: model specification to define an objective function
    :param flip_score: if `True` then multiple by -1. This is used for converting IC50 to sensitivity score.
    :param seed: random seed for parameter initialization
    :param save_interval: interval for saving results
    :param output_dir: output directory
    :returns: `parameters_trained` contains trained paramters and `output_dict` contains predictions
    """

    print ('Initializing the model ...')

    # Reset TensorFlow graph
    ops.reset_default_graph()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO: save the model configuration

    ################################
    ##### Setting up the model #####
    ################################

    # TODO: add other model types and update the scrip accordingly
    if model_spec_name not in ['cadrres', 'cadrres-wo-sample-bias']:
        return None

    n_drugs = train_resp_df.shape[1]
    drug_list = train_resp_df.columns

    n_samples = train_resp_df.shape[0]
    sample_list_train = train_resp_df.index
    sample_list_test = test_resp_df.index

    n_x_features = train_feature_df.shape[1]
    n_y_features = n_drugs

    X_train_dat = np.array(train_feature_df)
    Y_train_dat = np.identity(n_drugs)

    X_test_dat = np.array(test_feature_df)
    Y_test_dat = np.identity(n_drugs)

    ##### Scale (1-auc) value #####
    S_train_obs = np.array(train_resp_df) * 240 - 80
    S_test_obs = np.array(test_resp_df) * 240 - 80

    ##### Convert log(IC50) to sensitivity scores #####
    #if flip_score:
    #    S_train_obs = np.array(train_resp_df) * -1
    #    S_test_obs = np.array(test_resp_df) * -1
    #else:
    #    S_train_obs = np.array(train_resp_df)
    #    S_test_obs = np.array(test_resp_df)

    ##### Initialize placeholders and parameters #####
    X_train, Y_train = create_placeholders(n_x_features, n_y_features)
    parameters = initialize_parameters(n_samples, n_drugs, n_x_features, n_y_features, n_dim, seed)

    ##### Extract only prediction of only observed drug response #####
    train_known_idx = np.where(~np.isnan(S_train_obs.reshape(-1)))[0]
    n_train_known = len(train_known_idx)
    print ("Train:", len(train_known_idx), "out of", n_drugs * n_samples)

    S_train_pred = inward_propagation(X_train, Y_train, parameters, n_samples, n_drugs, model_spec_name)
    S_train_pred_resp = tf.gather(tf.reshape(S_train_pred, [-1]), train_known_idx, name="S_train_pred_resp")
    S_train_obs_resp = tf.convert_to_tensor(S_train_obs.reshape(-1)[train_known_idx], np.float32, name="S_train_obs_resp")

    #### Calculate the difference between the predicted sensitivity and the actual #####
    diff_op_train = tf.subtract(S_train_pred_resp, S_train_obs_resp, name="raw_training_error")

    with tf.name_scope("train_cost") as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op_train, name="squared_diff_train"), name="sse_train")
        regularizer = tf.multiply(tf.add(tf.reduce_sum(tf.square(parameters['W_P'])), tf.reduce_sum(tf.square(parameters['W_Q']))), lda, name="regularize")
        cost_train = tf.math.divide(tf.add(base_cost, regularizer), n_train_known * 2.0, name="avg_error_train")

    # TODO: add different kinds of regulalization (the current version uses ridge; see CaDRReS2_tf_matrix_factorization_wo_bp_lasso.py)
    
    ##### Use an exponentially decaying learning rate #####
    # learning_rate = tf.train.exponential_decay(l_rate, global_step, 10000, 0.96, staircase=True)

    ##################################################
    ##### Initialize session and train the model #####
    ##################################################

    print ('Starting model training ...')

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
        train_step = optimizer.minimize(cost_train, global_step=global_step)
        mse_summary = tf.summary.scalar("mse_train", cost_train)

    sess = tf.Session()

    # TODO: save model every save_interval
    # summary_op = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("{}/tf_matrix_factorization_logs".format(output_dir), sess.graph)

    sess.run(tf.global_variables_initializer())
    parameters_init = sess.run(parameters)

    cost_train_vals = []
    cost_test_vals = []

    start = time.time()
    for i in range(max_iter):    
        _ = sess.run(train_step, feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
        
        if i % save_interval == 0:

            # training step
            res = sess.run(cost_train, feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
            cost_train_vals += [res]

            time_used = (time.time() - start)
            print("MSE train at step {}: {:.3f} ({:.2f}m)".format(i, cost_train_vals[-1], time_used/60))

            # save parameter
            parameters_trained = sess.run(parameters)
            parameters_trained['sample_list_train'] = sample_list_train
            parameters_trained['sample_list_test'] = sample_list_test
            # make a prediction
            test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)

            cost_test_vals += [test_cost]
            # summary_str = res[0]
            # writer.add_summary(summary_str, i)

    parameters_trained, train_pred = sess.run([parameters, S_train_pred], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})
    parameters_trained['sample_list_train'] = sample_list_train
    parameters_trained['sample_list_test'] = sample_list_test

    test_pred, test_cost = predict(X_test_dat, Y_test_dat, S_test_obs, parameters_trained, sample_list_test, model_spec_name, False)
    train_pred, train_cost = predict(X_train_dat, Y_train_dat, S_train_obs, parameters_trained, sample_list_train, model_spec_name, True)
    parameters_trained['mse_train_vals'] = cost_train_vals
    parameters_trained['mse_test_vals'] = cost_test_vals

    P_train, Q_train = get_latent_vectors(X_train, Y_train, parameters)
    P, Q = sess.run([P_train, Q_train], feed_dict={X_train: X_train_dat, Y_train: Y_train_dat})

    sess.close()

    ############################
    ##### Saving the model #####
    ############################

    print ('Saving model parameters and predictions ...')

    # Save model configurations

    parameters_trained['n_dim'] = n_dim
    parameters_trained['lda'] = lda
    parameters_trained['max_iter'] = max_iter
    parameters_trained['l_rate'] = l_rate
    parameters_trained['model_spec_name'] = model_spec_name
    parameters_trained['seed'] = seed

    parameters_trained['drug_list'] = drug_list
    parameters_trained['train_sample_list'] = sample_list_train
    parameters_trained['kernel_sample_list'] = list(train_feature_df.columns)

    # Save the prediction and processed data

    pred_df = pd.DataFrame(test_pred, index=sample_list_test, columns=drug_list) * -1
    obs_df = pd.DataFrame(S_test_obs, index=sample_list_test, columns=drug_list) * -1

    pred_train_df = pd.DataFrame(train_pred, index=sample_list_train, columns=drug_list) * -1
    obs_train_df = pd.DataFrame(S_train_obs, index=sample_list_train, columns=drug_list) * -1

    output_dict = {}
    output_dict['pred_test_df'] = pred_df
    output_dict['obs_test_df'] = obs_df
    output_dict['pred_train_df'] = pred_train_df
    output_dict['obs_train_df'] = obs_train_df

    bq = parameters_trained['b_Q']
    bq_df = pd.DataFrame([drug_list, list(bq.flatten())]).T
    bq_df.columns = ['drug_name', 'drug_bias']
    bq_df = bq_df.set_index('drug_name')

    P_df = pd.DataFrame(P, index=sample_list_train, columns=range(1, n_dim+1))
    Q_df = pd.DataFrame(Q, index=drug_list, columns=range(1, n_dim+1))

    output_dict['b_Q_df'] = bq_df
    output_dict['P_df'] = P_df
    output_dict['Q_df'] = Q_df

    print ('DONE')

    return parameters_trained, output_dict



def get_sample_weights_logistic_x0(drug_df, log2_max_conc_col_name, sample_list):

    """
    Calculate weights_logistic_x0_df, which is an input of train_model_logistic_weight. The logistic weight is assigned to each drug-sample pair with respect to maximum drug dosage.
    """

    drug_list = drug_df.index
    max_conc = np.array(drug_df[[log2_max_conc_col_name]])
    n_samples = len(sample_list)
    weights_logistic_x0 = np.repeat(max_conc.T, n_samples, axis=0)
    weights_logistic_x0_df = pd.DataFrame(weights_logistic_x0, columns=drug_list, index=sample_list)

    return weights_logistic_x0_df

if __name__ == "__main__":
    datadir = '/Users/yihyun/Code/'
    # import data
    ## fearture genes
    with open ("/volume/yihyun/drug/MF_model/feature_genes.txt", "r") as f:
        feature_genes = f.read().split("\n")
    f.close()
    ## ccle gene expression
    ccle_exp = pd.read_csv("/volume/yihyun/drug/CCLE_expression.csv", index_col = 0)
    ### ccle preprocess
    col_name = []
    for name in list(ccle_exp.columns):
        #print(name.split(" "))
        col_name.append(name.split(" ")[0])
    ccle_exp.columns = col_name
    ## drug response data
    prism_train = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_train.csv"), index_col = 'smiles')
    prism_test = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test.csv"), index_col = 'smiles')
    prism_test_cl = pd.read_csv(os.path.join(datadir, "drug_sensitivity_prediction/0.data/processed/prism_test_scdrug.csv"), index_col = 'smiles')

    # cell line kernel calculation
    log_feature_exp = np.log2(ccle_exp + 1).T
    logFC_exp = (log_feature_exp.T - log_feature_exp.mean(axis=1)).T
    all_kernel = calculate_kernel_feature(logFC_exp,logFC_exp,feature_genes)
    all_kernel.to_csv(os.path.join(datadir, "drug_sensitivity_prediction/2.Combined_model_evaluation/2_1.CaDRReS_CLsim/kernel_cl_train.csv"), index = True)


    # kernel feature based only on training samples
    cell_line_sample_list = prism_train.columns.tolist()
    X_train = all_kernel.loc[cell_line_sample_list, cell_line_sample_list]
    print(X_train.shape)
    # observed drug response
    #Y_train = prism_train.T.loc[cell_line_sample_list] # cell-blind set prediction
    Y_train = pd.concat([prism_train, prism_test], join='inner').T.loc[cell_line_sample_list] # cell-blind set + disjoint set prediction
    print(Y_train.shape)

    # specify output directry
    output_dir = './save_model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print ('Results will be saved in ', output_dir)

    # train model
    cadrres_model_dict, cadrres_output_dict = train_model(Y_train, X_train, Y_train, X_train, 10, 0.0, 150000, 0.01, 
                                                          model_spec_name="cadrres-wo-sample-bias", save_interval=5000, 
                                                          output_dir=output_dir)
    # save model
    ## cell-blind set prediction
    # print('Saving ' + output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLSim"))
    # pickle.dump(cadrres_model_dict, open(output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLSim"), 'wb'))
    # print('Saving ' + output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLSim"))
    # pickle.dump(cadrres_output_dict, open(output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLSim"), 'wb'))

    ## cell-blind set + disjoint set prediction
    print('Saving ' + output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLSim_allMol"))
    pickle.dump(cadrres_model_dict, open(output_dir + '{}_param_dict.pickle'.format("cadrres-wo-sample-bias_CLSim_allMol"), 'wb'))
    print('Saving ' + output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLSim_allMol"))
    pickle.dump(cadrres_output_dict, open(output_dir + '{}_output_dict.pickle'.format("cadrres-wo-sample-bias_CLSim_allMol"), 'wb'))



