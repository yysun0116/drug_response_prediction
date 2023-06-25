#%%
import sys
import timeit
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score

import preprocess as pp
import os
import random
#%%
class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 452)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        """Classifier or regressor based on multilayer perceptron."""
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

        inputs = data_batch[:-2]
        mask = torch.stack(data_batch[-2], dim = 0)
        correct_values = torch.stack(data_batch[-1], dim = 0)

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.view(-1, 452)
            #loss = F.mse_loss(predicted_values, correct_values)
            loss = self.masked_mse(predicted_values, correct_values, mask)
            # loss = self.masked_mse(predicted_values, correct_values, mask) - self.masked_aCC(predicted_values, correct_values, mask)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
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

    (file_dir, task, dataset, radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = sys.argv[1:]
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [radius, dim, layer_hidden, layer_output,
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

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    data_train, dataset_test, N_fingerprints, fingerprint_dict = pp.create_datasets(file_dir, radius, device, norm=False)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            N_fingerprints, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = './output/result--multi-output' + setting + '.txt'
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_val\taCC_val\tMAE_test\taCC_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')


    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join('runs/GNN_multi-output'+ dataset))
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
    torch.save(model.state_dict(), '/volume/yihyun/drug/baseline_model/save_model/molecularGNN_smiles_multi-output.pt')
    print("Procedure done!")
