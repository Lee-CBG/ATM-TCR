import os
import re
import sys
import time
import math
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def cuda(tensor, is_cuda):
        
    if is_cuda : return tensor.cuda()
    else : return tensor

def idxtobool(idx, size, is_cuda):
    
    V = cuda(torch.zeros(size, dtype = torch.float), is_cuda)
    if len(size) > 2:
        
        for i in range(size[0]):
            for j in range(size[1]):
                subidx = idx[i, j, :]
                V[i, j, subidx] = float(1)
    elif len(size) is 2:
        
        for i in range(size[0]):
            subidx = idx[i,:]
            V[i,subidx] = float(1)
            
    else:
        raise argparse.ArgumentTypeError('len(size) should be larger than 1')
            
    return V

def create_tensorboard(tensorboard_name):

    tbf = None
    summary_dir = Path('tensorboards').joinpath(tensorboard_name)
    if not summary_dir.exists(): summary_dir.mkdir(parents=True)
    tbf = SummaryWriter(log_dir = str(summary_dir))
    
    return tbf

def get_performance_fidelity_batchiter(loader, model, prior, args, device='cpu'):

    model.eval()

    K = args.K
    beta = args.beta
    is_cuda = args.cuda
    batch_size = args.batch_size
    class_criterion = nn.CrossEntropyLoss(reduction = 'sum')
    info_criterion = nn.KLDivLoss(reduction = 'sum')
    
    total_loss_cont, class_loss_cont, info_loss = 0, 0, 0
    total_loss_fixed, class_loss_fixed = 0, 0
    score_cont_all, score_fixed_all, label = [], [], []
    for batch in loader:
        
        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        #score, p_pep, p_tcr, z_pep, z_tcr, score_fixed = model(X_pep, X_tcr, args.num_sample) 
        score_cont, p_pep, p_tcr, _, _, score_fixed = model(X_pep, X_tcr, args.num_sample)
        
        p_pep_prior = cuda(prior(var_size=p_pep.size()), is_cuda)
        p_tcr_prior = cuda(prior(var_size=p_tcr.size()), is_cuda)

        class_loss_fixed += class_criterion(score_fixed, y).div(math.log(2)).item() / batch_size
        class_loss_cont += class_criterion(score_cont, y).div(math.log(2)).item() / batch_size
        info_loss += K * (info_criterion(p_pep, p_pep_prior).item()+info_criterion(p_tcr, p_tcr_prior).item()) / batch_size
        total_loss_fixed += class_loss_fixed + beta * info_loss
        total_loss_cont += class_loss_cont + beta * info_loss
        
        score_fixed_all.extend(score_fixed.data.cpu().tolist())
        score_cont_all.extend(score_cont.data.cpu().tolist())
        label.extend(y.data.cpu().tolist())

    perf_fixed = get_performance(score_fixed_all, label)
    perf_fixed['class_loss'] = round(class_loss_fixed, 4)
    perf_fixed['info_loss'] = round(info_loss, 4)
    perf_fixed['total_loss'] = round(total_loss_fixed, 4)    
    perf_cont = get_performance(score_cont_all, label)
    perf_cont['class_loss'] = round(class_loss_cont, 4)
    perf_cont['info_loss'] = round(info_loss, 4)
    perf_cont['total_loss'] = round(total_loss_cont, 4)

    return perf_fixed, perf_cont


def write_explain_batchiter(loader, model, wf, device='cpu'):

    model.eval()
    
    rev_peploader = loader['pep_amino_idx']
    rev_tcrloader = loader['tcr_amino_idx']
    loader = loader['loader']
    for batch in loader:
        
        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        _, p_pep, p_tcr, _, _, score = model(X_pep, X_tcr)
        p_pep = np.exp(p_pep.data.cpu().tolist())
        p_tcr = np.exp(p_tcr.data.cpu().tolist())
        score = score.data.cpu().tolist()
        pred = np.argmax(score, -1)
        
        for i in range(len(pred)):

            pep_seq = ''.join([rev_peploader[x] for x in X_pep[i]])
            pep_seq = re.sub(r'<pad>', '', pep_seq)
            tcr_seq = ''.join([rev_tcrloader[x] for x in X_tcr[i]])
            tcr_seq = re.sub(r'<pad>', '', tcr_seq)

            newrow = [int(y[i]), int(pred[i]), float(np.exp(score[i][1]))]
            newrow.extend([pep_seq])
            newrow.extend(p_pep[i])
            newrow.extend([tcr_seq])
            newrow.extend(p_tcr[i])
            wf.writerow(newrow)

                
def write_blackbox_output_batchiter(loader, model, wf, device='cpu', ifscore=False):

    model.eval()
    
    rev_peploader = loader['pep_amino_idx']
    rev_tcrloader = loader['tcr_amino_idx']
    loader = loader['loader']
    for batch in loader:
        
        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        score = model(X_pep, X_tcr).data.cpu().tolist()
        pred = np.argmax(score, -1)
        
        for i in range(len(pred)):

            pep_seq = ''.join([rev_peploader[x] for x in X_pep[i]])
            pep_seq = re.sub(r'<pad>', '', pep_seq)
            tcr_seq = ''.join([rev_tcrloader[x] for x in X_tcr[i]])
            tcr_seq = re.sub(r'<pad>', '', tcr_seq)
            if ifscore:
                wf.writerow([pep_seq, tcr_seq, int(y[i]), int(pred[i]), float(np.exp(score[i][1]))])
            else:
                wf.writerow([pep_seq, tcr_seq, int(pred[i])])
    

def get_performance_batchiter(loader, model, device='cpu'):
    '''
    print classification performance for binary task

    Args:
     loader  - data loader
     model   - classification model
     loss_ft - loss function
    '''
    model.eval()
    
    loss = 0
    score, label = [], []
    for batch in loader:
        
        X_pep, X_tcr, y = batch.X_pep.to(device), batch.X_tcr.to(device), batch.y.to(device)
        yhat = model(X_pep, X_tcr)
        loss += F.cross_entropy(yhat, y, reduction='sum').item()
        #score.extend(np.ndarray.flatten(yhat.data.cpu().numpy()))
        #label.extend(np.ndarray.flatten(y.data.cpu().numpy()))
        score.extend(yhat.data.cpu().tolist())
        label.extend(y.data.cpu().tolist())

    perf = get_performance(score, label)
    perf['loss'] = round(loss, 4)

    return perf

def get_performance(score, label):
    '''
    get classification performance for binary task

    Args:
     score - 1D np.array or list
     label - 1D np.array or list
    '''

    accuracy = None
    precision1, precision0 = None, None
    recall1, recall0 = None, None
    f1macro, f1micro = None, None
    auc = None

    #if type(score) is list():
    #    score = np.array(score)
    #if type(label) is list():
    #    label = np.array(label)

    label_pred = np.argmax(score, -1)
    accuracy = accuracy_score(label, label_pred)
    precision1 = precision_score(label, label_pred, pos_label=1)
    precision0 = precision_score(label, label_pred, pos_label=0)
    recall1 = recall_score(label, label_pred, pos_label=1)
    recall0 = recall_score(label, label_pred, pos_label=0)
    f1macro = f1_score(label, label_pred, average='macro')
    f1micro = f1_score(label, label_pred, average='micro')
    auc = roc_auc_score(np.array(label), np.array(score)[:,-1]) if len(np.unique(np.array(label)))!=1 else -1

    ndigits = 4
    performance = {'accuracy': round(accuracy, ndigits),
                   'precision1': round(precision1, ndigits), 'precision0': round(precision0, ndigits),
                   'recall1': round(recall1, ndigits), 'recall0': round(recall0, ndigits),
                   'f1macro': round(f1macro, ndigits), 'f1micro': round(f1micro, ndigits),
                   'auc': round(auc, ndigits)}
    tn,fp,fn,tp = confusion_matrix(label, label_pred).ravel()
    print(tn,fp,fn,tp)
    
    return performance

def print_performance(perf, printif=True, writeif=False, boardif=False, **kargs):
    '''
    print classification performance for binary task

    Args:
     per   - dictionary with measure name as keys and performance as values 
             or perf = get_performance(score, label)
     kargs - epoch, loss, global_step
             wf = open(outfile_name, 'w')
             tbf = create_tensorboard(tensorboard_name)
    '''

    measures = sorted(perf.keys())

    if printif:
        maxchrlen = max([len(x) for x in measures])
        for mea in measures:
            print(mea + ' ' * (maxchrlen - len(mea)) + ' {:.4f}'.format(perf[mea]))
        print('')

    if boardif:
        assert 'tbf' in kargs.keys(), 'missing argument: tbf'
        assert 'global_step' in kargs.keys(), 'missing argument: global_step'
        assert 'mode' in kargs.keys(), 'missing argument: mode'
        for mea in measures:
            kargs['tbf'].add_scalars(main_tag='performance/{}'.format(mea),
                            tag_scalar_dict={kargs['mode']:perf[mea]},
                            global_step=kargs['global_step'])

    if writeif:
        assert 'wf' in kargs.keys(), 'missing argument: wf'
        #newrow = [perf[x] for x in measures]
        #kargs['wf'].writerow(newrow)
        kargs['wf'].writerow(perf)
        return kargs['wf']

            
def str2bool(v):
    """
    Convert string to boolean object
    
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True', 'Y', 'Yes', 'YES', 'YEs', 'ye'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False', 'N', 'NO', 'No'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
def timeSince(since):
    """
    Credit: https://github.com/1Konny/VIB-pytorch
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    
    return '%dm %ds' % (m, s)


def check_model_name(model_name, file_path = 'models'):
    """
    Check whether model name is overlapped or not 
    """
    if model_name in os.listdir(file_path):
        
        valid = {"yes": True, "y": True, "ye": True, 'true': True, 't': True, '1': True, "no": False, "n": False, 'false': False, 'f': False, '0': False}
        sys.stdout.write("The file {} already exists. Do you want to overwrite it? [yes/no]".format(model_name))
        choice = input().lower()
    
        if choice in valid:
            if not valid[choice]:
                sys.stdout.write("Please assign another name. (ex. 'original_2.ckpt')")
                model_name = input().lower()
                check_model_name(model_name = model_name, file_path = file_path)
                
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            check_model_name(model_name = model_name, file_path = file_path)
            
    return model_name
