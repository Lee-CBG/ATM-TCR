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

#import re
#import math
#import copy
#import _pickle as pickle
#import torch
#from PIL import Image
#from PIL import ImageDraw
#from torch import nn
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import label_binarize

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
        choice = raw_input().lower()
    
        if choice in valid:
            if not valid[choice]:
                sys.stdout.write("Please assign another name. (ex. 'original_2.ckpt')")
                model_name = raw_input().lower()
                check_model_name(model_name = model_name, file_path = file_path)
                
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            check_model_name(model_name = model_name, file_path = file_path)
            
    return model_name
'''
def compare_with_previous(correct, correct_pre, compare = True):
    
    if compare and correct < correct_pre:
        valid = {"yes": True, "y": True, "ye": True, 'true': True, 't': True, '1': True, "no": False, "n": False, 'false': False, 'f': False, '0': False}
        sys.stdout.write('the accuracy is not improved from the last model. Do you want to proceed it? [yes/no]')
        choice = input().lower()
        
        if choice in valid:
            if not valid[choice]: sys.exit(0)
        
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            compare_with_previous(correct = correct, correct_pre = correct_pre, compare = compare)

def get_selected_words(x_single, score, id_to_word, k): 
    
    #selected_words = {} # {location: word_id}
    selected = np.argsort(score)[-k:] 
    selected_k_hot = np.zeros(400)
    selected_k_hot[selected] = 1.0

    x_selected = (x_single * selected_k_hot).astype(int)
    return x_selected 

def create_dataset_from_score(x, scores, k):
    
    with open('data/id_to_word.pkl','rb') as f:
        id_to_word = pickle.load(f)
    new_data = []
    #new_texts = []
    for i, x_single in enumerate(x):
        x_selected = get_selected_words(x_single, 
            scores[i], id_to_word, k)

        new_data.append(x_selected) 

    np.save('data/x_val-L2X.npy', np.array(new_data))


def label2binary(label, classes):
    classes = list(classes)
    if len(classes) == 2:
        classes.append(-1)
        res = label_binarize(label, classes = classes)
        return res[:, :-1]
    else:
        return label_binarize(label, classes = classes)
        

def query_yes_no(question, default = "yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credit: fmark and Nux
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, 'true': True, 't': True, '1': True,
             "no": False, "n": False, 'false': False, 'f': False, '0': False}
    
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")

class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)

class save_batch(object):

    def __init__(self, dataset, batch, label, label_pred, label_approx, index, filename, is_cuda, word_idx = None): #change
        
        self.batch = batch
        self.label = label
        self.label_pred = label_pred
        self.label_approx = label_approx
        self.index = index
        self.filename = filename
        self.is_cuda = is_cuda
        self.word_idx = word_idx
        func = getattr(self, dataset)
        self.output = func()

    def imdb(self):

        word_idx = self.word_idx          
        width = 100
        #height = 100
        current_line = 0 
        skip_line = 10
        if len(self.index.size()) == 3:
            self.index = self.index.view(self.index.size(0), self.index.size(1) * self.index.size(2))
        
        img = Image.new("RGBA", (700, 10000), 'white')
        draw = ImageDraw.Draw(img)
        
        for i in range(self.batch.size(0)):
            
            current_line = current_line + skip_line * 2
            label_review = "POS" if self.label[i].item() == 1 else "NEG"
            label_review_pred = "POS" if self.label_pred[i].item() == 1 else "NEG"
            label_review_approx = "POS" if self.label_approx[i].item() == 1 else "NEG"
            review = idxtoreview(review = self.batch[i], word_idx = word_idx)
            review_selected = idxtoreview(review = self.batch[i], word_idx = word_idx, index = self.index[i])
            draw.text((20, current_line), "sent: " + label_review + " pred:" + label_review_pred + " approx:" + label_review_approx, 'blue')
           
            num_line = len(review) // width + 1
            
            for j in range(num_line):
                
                current_line = current_line + skip_line
                draw.text((20, current_line), review[(width * j):(width * (j + 1))], 'black')    
                draw.text((20, current_line), review_selected[(width * j):(width * (j + 1))], 'red')
                
        draw = ImageDraw.Draw(img)            
        img.save(str(self.filename))

        ## write the selected chunks only
        textfile = open(str(self.filename.with_suffix('.txt')), 'w')
        for i in range(self.batch.size(0)):
            
            label_review = "POS" if self.label[i].item() == 1 else "NEG"
            label_review_pred = "POS" if self.label_pred[i].item() == 1 else "NEG"
            label_review_approx = "POS" if self.label_approx[i].item() == 1 else "NEG" 
            textfile.write("sent: " + label_review + " pred:" + label_review_pred + " approx:" + label_review_approx)
            
            review_selected = idxtoreview(review = self.batch[i], word_idx = word_idx, index = self.index[i])
            textfile.write(review_selected)
        
        textfile.close()
        
        
    def mnist(self):
        """
        draw and save MNIST images and selected chunks
        """
        img = copy.deepcopy(self.batch)    
        n_img = img.size(0)
        n_col = 8
        n_row = n_img // n_col + 1

        fig = plt.figure(figsize=(n_col * 1.5, n_row * 1.5)) 

        for i in range(n_img):

            plt.subplot(n_row, n_col, 1 + i)
            plt.axis('off')
            # original image
            img0 = img[i].squeeze(0)#.numpy()
            plt.imshow(img0, cmap = 'autumn_r')
            # chunk selected
            img2 = img[i].view(-1)#.numpy()
            img2[self.index[i]] = cuda(torch.tensor(float('nan')), self.is_cuda)
            img2 = img2.view(img0.size())#.numpy()
            plt.title('BB {}, Apx {}'.format(self.label_pred[i], self.label_approx[i]))
            plt.imshow(img2, cmap = 'gray')

        fig.subplots_adjust(wspace = 0.05, hspace = 0.35)      
        fig.savefig(str(self.filename))

def idxtoreview(review, word_idx, index = None):
    
    review = np.array(word_idx)[review.tolist()]
    review = [re.sub(r"<pad>", "", review_sub) for review_sub in review]
    review = [re.sub(' +', ' ', review_sub) for review_sub in review]
    review = [review_sub.strip() for review_sub in review]
    
    if index is not None:    

        review_selected = [len(review_sub) * "_" for review_sub in review]
        for index_sub in index:
            review_selected[index_sub] = review[index_sub]
        review = review_selected
    
    review = " ".join(review)
    review = re.sub(' +', ' ', review)
    review = review.strip()
    
    return review

class index_transfer(object):

    def __init__(self, dataset, idx, filter_size, original_ncol, original_nrow, is_cuda = False):
        
        self.dataset = dataset
        self.idx = idx
        if type(filter_size) is int:
            filter_size = (filter_size, filter_size)
        self.filter_size_row = filter_size[0]
        self.filter_size_col = filter_size[1]
        self.original_ncol = original_ncol
        self.original_nrow = original_nrow
        self.is_cuda = is_cuda
        func = getattr(self, dataset)
        self.output = func()
    
    def default(self): 

        assert  self.original_nrow % self.filter_size_row < 1
        assert  self.original_ncol % self.filter_size_col < 1
        bat_size = self.idx.size(0)
        ncol = cuda(torch.LongTensor([self.original_ncol // self.filter_size_col]), self.is_cuda)
        #nrow = cuda(torch.LongTensor([self.original_nrow // self.filter_size_row]), self.is_cuda)
        #idx_2d = torch.stack([torch.div(self.idx, ncol), torch.remainder(self.idx, ncol)])
        
        idx_2d_unpool0 = torch.add(torch.mul(torch.div(self.idx, ncol), self.filter_size_row).view(-1, 1), cuda(torch.arange(self.filter_size_row), self.is_cuda)).view(-1, self.filter_size_row)
        idx_2d_unpool1 = torch.add(torch.mul(torch.remainder(self.idx, ncol), self.filter_size_col).view(-1, 1), cuda(torch.arange(self.filter_size_col), self.is_cuda)).view(-1, self.filter_size_col)

        idx_2d_unpool0 = idx_2d_unpool0.view(-1, 1).expand(-1, self.filter_size_col).contiguous().view(bat_size, -1)
        idx_2d_unpool1 = idx_2d_unpool1.view(-1, 1).expand(-1, self.filter_size_row).contiguous().view(bat_size, -1)
        
        idx_2d_unpool0 = torch.mul(idx_2d_unpool0, cuda(torch.LongTensor([self.original_ncol]), self.is_cuda))
        idx_2d_unpool1 = torch.transpose(idx_2d_unpool1.view(-1, self.filter_size_row, self.filter_size_col), 1, 2).contiguous().view(bat_size, -1)
        
        idx_unpool = torch.add(idx_2d_unpool0, idx_2d_unpool1)
        
        return idx_unpool
    
    def imdb(self):

        if self.filter_size_col < self.original_ncol:
            
            chunk_size = self.filter_size_col
            newadd = cuda(torch.LongTensor(range(chunk_size)), self.is_cuda).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            new_size_col = self.original_ncol - chunk_size + 1
            self.idx = torch.add(self.idx, torch.mul(torch.div(self.idx, new_size_col), chunk_size - 1))
            self.idx = torch.add(self.idx.unsqueeze(-1).expand(-1,-1,-1,chunk_size), newadd)
            newsize = self.idx.size()
            self.idx = self.idx.view(newsize[0], newsize[1], -1, 1).squeeze(-1)
            
            return self.idx
        
        else:

            return self.default()
        
    def mnist(self):    
        
        return self.default()
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)  
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def Concatenate(input_global, input_local):
    
    input_global = input_global.unsqueeze(-2)
    input_global = input_global.expand(-1, input_local.size(-2), -1)
            
    return torch.cat((input_global, input_local), -1)

class TimeDistributed(nn.Module):
    
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):

        assert (len(x.size()) <= 3)

        if len(x.size()) <= 2:
            return self.module(x)

        x = x.permute(0, 2, 1) # reshape x
        y = self.module(x)
        
        if len(y.size()) == 3:
            y = y.permute(0, 2, 1) # reshape y

        return y
'''
