#%%
import sys
import timeit
import math

import numpy as np
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

import os
import random

#%%
class SMILESVec_mlp(nn.Module):
    def __init__(self, dim, layer_output):
        super(SMILESVec_mlp, self).__init__()
        self.W_input = nn.Linear(100, dim)
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 452)


    def mlp(self, vectors):
        vectors = torch.relu(self.W_input(vectors))
        """regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    
    def masked_mse(self, pred, label, mask):
        # multi-output mse with masking
        return torch.sum(((pred-label)**2)*mask)/torch.sum(mask)

    def masked_aCC(self, pred, label, mask):
        # multi-output average correlation coefficient with masking
        d_label = (label - (torch.sum(label*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
        d_pred = (pred - (torch.sum(pred*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
        x = torch.sum(d_label*d_pred, 1)
        y = torch.sqrt(torch.sum(d_label**2, 1) * torch.sum(d_pred**2, 1))
        aCC = torch.mean(x/y)
        return aCC

    def forward_regressor(self, data_batch, train):

        inputs = torch.cat(data_batch[0]).view(-1, 100).to(device)
        mask = torch.stack(data_batch[-2], dim = 0).to(device)
        correct_values = torch.stack(data_batch[-1], dim = 0).to(device)

        if train:
            predicted_values = self.mlp(inputs)
            predicted_values = predicted_values.view(-1, 452)
            loss = self.masked_mse(predicted_values, correct_values, mask)
            return loss
        else:
            with torch.no_grad():
                predicted_values = self.mlp(inputs)
            predicted_values = predicted_values.view(-1, 452)
            return predicted_values, correct_values, mask


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        batch = 0
        for i in range(0, N, batch_train):
            batch += 1
            data_batch = list(zip(*dataset[i:i+batch_train]))

            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

            writer.add_scalar("training loss", 
                        loss_total,
                        epoch*round(N/batch_train)+batch)
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset):
        N = len(dataset)
        batch = 0
        SAE = 0
        mask_sum = 0
        CC_ls = torch.tensor([]).to(device)
        for i in range(0, N, batch_test):
            batch += 1
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values, mask = self.model.forward_regressor(
                                               data_batch, train=False)

            SAE += self.masked_SAE(predicted_values, correct_values, mask)
            mask_sum += torch.sum(mask)
            CC = self.masked_aCC(predicted_values, correct_values, mask)
            CC_ls = torch.cat((CC_ls, CC),0)
            aCC_i = torch.mean(CC).to('cpu').data.numpy()
            writer.add_scalar("testing PCC", 
                        aCC_i,
                        epoch*round(N/batch_test)+batch)
        MAE = SAE/mask_sum
        MAE = MAE.to('cpu').data.numpy()
        aCC = torch.mean(CC_ls).cpu().data.numpy()
        return MAE, aCC

    def masked_SAE(self, pred, label, mask):
        return torch.sum(torch.abs(pred-label)* mask)

    def masked_aCC(self, pred, label, mask):
        # multi-output average correlation coefficient with masking
        d_label = (label - (torch.sum(label*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
        d_pred = (pred - (torch.sum(pred*mask,1)/torch.sum(mask, 1)).view(-1,1))*mask
        x = torch.sum(d_label*d_pred, 1)
        y = torch.sqrt(torch.sum(d_label**2, 1) * torch.sum(d_pred**2, 1))
        #aCC = torch.mean(x/y).to('cpu').data.numpy()
        return x/y

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

if __name__ == "__main__":

    (dataset, dim, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = sys.argv[1:]
    (dim, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [dim, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Import data')
    print('Just a moment......')
    # import preprocessed vectors of smiles
    with open("/volume/yihyun/drug/SMILESVec_model/data/smiles_train.vec", "rb") as f:
        train_vec = pickle.loads(f.read())
    with open("/volume/yihyun/drug/SMILESVec_model/data/smiles_test.vec", "rb") as f:
        test_vec = pickle.loads(f.read())
    # import auc data
    test_set = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_test.csv", index_col="smiles")
    train_set = pd.read_csv("/volume/yihyun/drug/baseline_model/data/prism_train.csv", index_col="smiles")
    # define mask
    mask_df_test = (test_set.isna() == False).astype(int)
    mask_df_train = (train_set.isna() == False).astype(int)
    # filling missing value
    test_set = test_set.fillna(value = -9)
    train_set = train_set.fillna(value = -9)
    # combine data
    dataset_test = []
    for i in range(len(test_set)):
        dataset_test.append((torch.FloatTensor(test_vec[i]), 
        torch.LongTensor(mask_df_test.loc[mask_df_test.index[i]]), torch.FloatTensor(test_set.loc[test_set.index[i]])))
    import torch
    data_train = []
    for i in range(len(train_set)):
        data_train.append((torch.FloatTensor(train_vec[i]), 
        torch.LongTensor(mask_df_train.loc[mask_df_train.index[i]]), torch.FloatTensor(train_set.loc[train_set.index[i]])))


    print('Creating a model.')
    torch.manual_seed(1234)
    model = SMILESVec_mlp(dim, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = './output/result--SMILESVec_multi-output-mlp' + setting + '.txt'
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_val\taCC_val\tMAE_test\taCC_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')


    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join('runs/SMILESVec_multi-output-mlp'+ dataset))
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(data_train))
    #print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)
    print("start training...")
    start = timeit.default_timer()

    for epoch in range(iteration):
        epoch += 1
        # spilt data
        dataset_train, dataset_dev = split_dataset(data_train, 0.9)

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        dev_MAE, dev_acc = tester.test_regressor(dataset_dev)
        test_MAE, test_acc = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                    dev_MAE, dev_acc, test_MAE, test_acc]))
        tester.save_result(result, file_result)

        print(result)
    
    print("saving model...")
    torch.save(model.state_dict(), '/volume/yihyun/drug/SMILESVec_model/save_model/SMILESVec_multi-output-mlp.pt')
    print("Procedure done!")
