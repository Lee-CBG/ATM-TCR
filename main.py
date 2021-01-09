import os
import sys
import csv
import re
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader, load_embedding
from utils import timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter

import data_io_tf

sys.path.append('../')

PRINT_EVERY_EPOCH = 1


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch in train_loader:

        X_pep, X_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)

        optimizer.zero_grad()
        yhat = model(X_pep, X_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss = F.binary_cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))


def run(
    infile, indepfile=None, blosum='data/BLOSUM45',
    batch_size=100, model_name='original.ckpt',
    epoch=200, lr=0.001, cuda=True, seed=7405,
    mode='train', model='cnn',
    dropRate=0.3, hid=10, linearSize=16, filters=100,
    padding='mid', maxTcrLen=None, maxPepLen=None,
    folds=10, testFold=9, validationFold=0,
    save_model=False
):

    if mode == 'test':
        assert indepfile is not None, '--indepfile is missing!'
    assert testFold < folds, '--testFold should be smaller than --folds'
    assert validationFold < folds, '--validationFold should be smaller than --folds'
    assert validationFold != testFold, '--validationFold and --testFold should not be equal to each other'

    # cuda
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    cuda = (cuda and torch.cuda.is_available())
    device = torch.device('cuda' if cuda else 'cpu')

    # set random seed
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # embedding matrix
    blosumFlag = True
    if blosum == None:
        blosumFlag = False
        blosum = 'data/BLOSUM45'
    embedding = load_embedding(blosum)

    # read data
    X_pep, X_tcr, y = data_io_tf.read_pTCR(infile)
    y = np.array(y)

    n_total = len(y)
    n_test = int(round(n_total / folds))
    n_train = n_total - n_test

    indexfile = re.sub('.txt', '_shuffleIdx.txt', infile)
    if os.path.exists(indexfile):
        idx_shuffled = np.loadtxt(indexfile, dtype=np.int32)
    else:
        idx_shuffled = np.arange(n_total)
        np.random.shuffle(idx_shuffled)
        np.savetxt(indexfile, idx_shuffled, fmt='%d')

    if validationFold < 0:
        idx_test = idx_shuffled[testFold*n_test:(testFold+1)*n_test]
        idx_train = list(set(idx_shuffled).difference(set(idx_test)))
    else:
        idx_test_remove = idx_shuffled[testFold*n_test:(testFold+1)*n_test]
        idx_test = idx_shuffled[validationFold *
                                n_test:(validationFold+1)*n_test]
        idx_train = list(set(idx_shuffled).difference(
            set(idx_test)).difference(set(idx_test_remove)))

    # define dataloader

    train_loader = define_dataloader(X_pep[idx_train], X_tcr[idx_train], y[idx_train],
                                     maxPepLen, maxTcrLen,
                                     padding=padding,
                                     batch_size=batch_size, device=device)
    test_loader = define_dataloader(X_pep[idx_test], X_tcr[idx_test], y[idx_test],
                                    maxlen_pep=train_loader['pep_length'],
                                    maxlen_tcr=train_loader['tcr_length'],
                                    padding=padding,
                                    batch_size=batch_size, device=device)

    pep_length = train_loader['pep_length']
    tcr_length = train_loader['tcr_length']

    # define model
    if model == 'cnn':

        from cnn import Net
        model = Net(embedding, pep_length, tcr_length, dropRate,
                    hid, linearSize, blosumFlag).to(device)
    else:
        raise ValueError('unknown model name')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    # fit model
    if mode == 'train':

        wf_open = open(
            'result/perf_' + os.path.splitext(os.path.basename(model_name))[0] + '.csv', 'w')
        wf_colnames = ['loss', 'accuracy',
                       'precision1', 'precision0',
                       'recall1', 'recall0',
                       'f1macro', 'f1micro', 'auc']
        wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')

        t0 = time.time()
        lossArraySize = 10
        lossArray = deque([sys.maxsize], maxlen=lossArraySize)
        for epoch in range(1, epoch + 1):

            train(model, device, train_loader['loader'], optimizer, epoch)
            perf_test = get_performance_batchiter(
                test_loader['loader'], model, device)

            # Print performance
            if epoch % PRINT_EVERY_EPOCH == 0:
                print('[TEST ] {} ----------------'.format(epoch))
                print_performance(perf_test, printif=False,
                                  writeif=True, wf=wf)

            # Check for early stopping
            min_epoch = 125
            lossArray.append(perf_test['loss'])
            averageLossChange = sum(np.abs(np.diff(lossArray))) / lossArraySize
            if epoch > min_epoch and averageLossChange < 10:
                print('Early stopping at epoch {}'.format(epoch))
                break

        print(os.path.splitext(os.path.basename(model_name))[0])
        print(timeSince(t0))

        # evaluate and print independent-test-set performance
        if indepfile is not None:
            print('[INDEP] {} ----------------')
            perf_indep = get_performance_batchiter(
                indep_loader['loader'], model, device)

            wf_open = open('result/perf_' + os.path.splitext(os.path.basename(model_name))[0] + '_' +
                           os.path.basename(indepfile), 'w')
            wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')
            print_performance(perf_indep, writeif=True, wf=wf)

            wf_open1 = open('data/pred_' + os.path.splitext(os.path.basename(model_name))[0] + '_' +
                            os.path.basename(indepfile), 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                indep_loader, model, wf1, device, ifscore=True)

        # evaluate and print test-set performance
        print('[TEST ] {} ----------------'.format(epoch))
        perf_test = get_performance_batchiter(
            test_loader['loader'], model, device)
        print_performance(perf_test)

        if save_model:

            wf_open1 = open(
                'result/pred_' + os.path.splitext(os.path.basename(model_name))[0] + '.csv', 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                test_loader, model, wf1, device, ifscore=True)

            model_name = './models/' + \
                os.path.splitext(os.path.basename(model_name))[0] + '.ckpt'
            torch.save(model.state_dict(), model_name)
    elif mode == 'indeptest':

        model_name = model_name

        assert model_name in os.listdir('./models')

        model_name = './models/' + model_name
        model.load_state_dict(torch.load(model_name))

        # evaluate and print independent-test-set performance
        print('[INDEP] {} ----------------')
        perf_indep = get_performance_batchiter(
            indep_loader['loader'], model, device)
        print_performance(perf_indep)

        # write blackbox output
        wf_bb_open1 = open('result/pred_' + os.path.splitext(os.path.basename(model_name))[0] + '_' +
                           os.path.basename(indepfile), 'w')
        wf_bb1 = csv.writer(wf_bb_open1, delimiter='\t')
        write_blackbox_output_batchiter(
            indep_loader, model, wf_bb1, device, ifscore=True)

    elif mode == 'physchm':
        loader = define_dataloader(X_pep, X_tcr, y,
                                   maxPepLen, maxTcrLen,
                                   padding=padding,
                                   batch_size=batch_size, device=device)
        features = get_physchem_properties_batchiter(
            loader["loader"], maxPepLen, maxTcrLen)

        directory = Path('psychmproperties')
        if not directory.exists():
            directory.mkdir(parents=True)
        with open('psychmproperties/tcrProperties.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            attributes = ['pepName', 'tcrName', 'basicity',
                          'hydrophobicity', 'helicity', 'mutation_stability']
            writer.writerow(attributes)
            for i in features.keys():
                properties = features[i]
                for j in range(len(properties['tcr'])):
                    rowToAppend = [i]
                    for item in properties:
                        rowToAppend.append(properties[item][j])
                    writer.writerow(rowToAppend)

    else:
        print('\nError: "--mode train" or "--mode test" expected')
