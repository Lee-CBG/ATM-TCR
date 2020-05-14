import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader,load_embedding
from utils import check_model_name,timeSince,get_performance_batchiter,print_performance,write_blackbox_output_batchiter
import data_io_tf
from pathlib import Path

sys.path.append('../')

PRINT_EVERY_EPOCH = 1

def train(model, device, train_loader, optimizer, epoch):
    
    model.train()

    for batch in train_loader:

        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        optimizer.zero_grad()
        yhat = model(X_pep, X_tcr)
        loss = F.cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))

def run(
        infile, indepfile=None, blosum='data/BLOSUM50', 
        batch_size=50, model_name='original.ckpt', 
        epoch=200, lr=0.001, cuda=True, seed=7405, 
        mode='train', model='cnn'
    ):

    print('Prediction of TCR binding to peptide-MHC complexes')

    if mode is 'test':
        assert indepfile is not None, 'indepfile argument is missing'
        
    # Check if using cuda or cuda is available
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with cude=True")
    cuda = (cuda and torch.cuda.is_available()) 
    device = torch.device('cuda' if cuda else 'cpu')

    # Set random seed
    seed = seed
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed) if cuda else None

    # Embedding matrix
    embedding = load_embedding(blosum)
      
    # Read data
    X_pep, X_tcr, y = data_io_tf.read_pTCR(infile)

    # TODO: Isn't this already a numpy array
    # y = np.array(y)

    # Separate into training, validation, and testing data.
    n_total = len(y)
    n_train = int(round(n_total * 0.8))
    n_valid = int(round(n_total * 0.1))
    n_test = n_total - n_train - n_valid
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)
    idx_train, idx_valid, idx_test = idx_shuffled[:n_train], \
                                     idx_shuffled[n_train:(n_train + n_valid)], \
                                     idx_shuffled[(n_train + n_valid):]

    # Define the dataloader for each of the steps of training/testing
    train_loader = define_dataloader(X_pep[idx_train], X_tcr[idx_train], y[idx_train], None,
                                     None, None,
                                     batch_size=batch_size, device=device)
    valid_loader = define_dataloader(X_pep[idx_valid], X_tcr[idx_valid], y[idx_valid], None,
                                     maxlen_pep=train_loader['pep_length'],
                                     maxlen_tcr=train_loader['tcr_length'],
                                     batch_size=batch_size, device=device)
    test_loader = define_dataloader(X_pep[idx_test], X_tcr[idx_test], y[idx_test], None,
                                    maxlen_pep=train_loader['pep_length'],
                                    maxlen_tcr=train_loader['tcr_length'],
                                    batch_size=batch_size, device=device)
        
    ## Read the independent dataset
    if indepfile is not None:
        X_indep_pep, X_indep_tcr, y_indep = data_io_tf.read_pTCR(indepfile)
        y_indep = np.array(y_indep)
        indep_loader = define_dataloader(X_indep_pep, X_indep_tcr, y_indep, None,
                                         maxlen_pep=train_loader['pep_length'],
                                         maxlen_tcr=train_loader['tcr_length'],
                                         batch_size=batch_size, device=device)

    # Ensure using a convolutional neural network
    if model == 'cnn':
        from cnn import Net
    else:
        raise ValueError('unknown model name')
    
    # Define Model
    model = Net(embedding, train_loader['pep_length'], train_loader['tcr_length']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    # Fit Model        
    if mode == 'train' : 

        # Validate File Name for Model    
        model_name = check_model_name(model_name)
        model_name = check_model_name(model_name, './models')
        model_name = model_name

        # Create CSV for Results
        wf_open = open('result/'+os.path.splitext(os.path.basename(infile))[0]+'_'+os.path.splitext(os.path.basename(model_name))[0]+'_valid.csv', 'w')
        wf_colnames = ['loss', 'accuracy',
                       'precision1', 'precision0',
                       'recall1', 'recall0',
                       'f1macro','f1micro', 'auc']
        wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')

        # Train the Model
        t0 = time.time()
        for epoch in range(1, epoch + 1):
            
            train(model, device, train_loader['loader'], optimizer, epoch)

            # Evaluate Performance
            perf_train = get_performance_batchiter(train_loader['loader'], model, device)
            perf_valid = get_performance_batchiter(valid_loader['loader'], model, device)

            # Print Training and Validation Performance
            print('Epoch {} TimeSince {}\n'.format(epoch, timeSince(t0)))
            print('[TRAINING] {} ----------------'.format(epoch))
            print_performance(perf_train)
            print('[VALIDATION] {} ----------------'.format(epoch))
            print_performance(perf_valid, writeif=True, wf=wf)

        # Evaluate Testing Performance
        print('[TESTING] {} ----------------'.format(epoch))
        perf_test = get_performance_batchiter(test_loader['loader'], model, device)
        print_performance(perf_test)

        model_name = './models/' + model_name
        torch.save(model.state_dict(), model_name)
            
    # Test Exisiting Model
    elif mode == 'test': 
        
        model_name = model_name

        assert model_name in os.listdir('./models')
        
        model_name = './models/' + model_name
        model.load_state_dict(torch.load(model_name))

        # Evaluate Performance on Independent Testing Data
        print('[INDEPENDENT TESTING] {} ----------------') 
        perf_indep = get_performance_batchiter(indep_loader['loader'], model, device)
        print_performance(perf_indep)

        # Write Blackbox Output
        wf_bb_open = open('data/testblackboxpred_' + os.path.basename(indepfile), 'w')
        wf_bb = csv.writer(wf_bb_open, delimiter='\t')
        write_blackbox_output_batchiter(indep_loader, model, wf_bb, device)

        wf_bb_open1 = open('data/testblackboxpredscore_' + os.path.basename(indepfile), 'w')
        wf_bb1 = csv.writer(wf_bb_open1, delimiter='\t')
        write_blackbox_output_batchiter(indep_loader, model, wf_bb1, device, ifscore=True)
        
    else:
        print('\nError: "mode=train" or "mode=test" expected')