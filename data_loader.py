import re
import torch
import numpy as np
#import data_io_tf
from torchtext import data

AMINO_MAP = {'<pad>':24, '*': 23, 'A': 0, 'C': 4, 'B': 20,
             'E': 6, 'D': 3, 'G': 7, 'F': 13, 'I': 9, 'H': 8,
             'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5, 'P': 14,
             'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
             'X': 22, 'Z': 21}
AMINO_MAP_REV = ['A','R','N','D','C','Q','E','G','H','I','L','K',
                 'M','F','P','S','T','W','Y','V','B','Z','X','*','<pad>']

def define_dataloader(X_pep, X_tcr, y, yhat=None, 
                    maxlen_pep=None, maxlen_tcr=None, 
                    padding='front',
                    batch_size=50, device='cuda'):

    device0 = 0 if device == 'cuda' else -1
    if maxlen_pep is None: maxlen_pep=max([len(x) for x in X_pep])
    if maxlen_tcr is None: maxlen_tcr=max([len(x) for x in X_tcr])
    if padding == 'front':
      pad_first_ = True
    elif padding == 'end':
      pad_first_ = False
    else:
      pad_first_ = False

    # define field
    field_pep = data.Field(tokenize=tokenizer, batch_first=True, 
                            pad_first=pad_first_, fix_length=maxlen_pep)
    field_tcr = data.Field(tokenize=tokenizer, batch_first=True, 
                            pad_first=pad_first_, fix_length=maxlen_tcr)
    field_y = data.Field(sequential=False, use_vocab=False)
    field_yhat = data.Field(sequential = False, use_vocab = False) if yhat is not None else None

    # define vocab
    amino_map = AMINO_MAP
    amino_map_rev = AMINO_MAP_REV
    '''
    amino_map = {'<pad>':24, '*': 23, 'A': 0, 'C': 4, 'B': 20,
                 'E': 6, 'D': 3, 'G': 7, 'F': 13, 'I': 9, 'H': 8,
                 'K': 11, 'M': 12, 'L': 10, 'N': 2, 'Q': 5, 'P': 14,
                 'S': 15, 'R': 1, 'T': 16, 'W': 17, 'V': 19, 'Y': 18,
                 'X': 22, 'Z': 21}
    amino_map_rev = ['A','R','N','D','C','Q','E','G','H','I','L','K',
                     'M','F','P','S','T','W','Y','V','B','Z','X','*','<pad>']
    '''
    field_pep.build_vocab()
    field_tcr.build_vocab()
    field_y.build_vocab()
    field_yhat.build_vocab() if yhat is not None else None
    field_pep.vocab.stoi = amino_map
    field_tcr.vocab.stoi = amino_map
    field_pep.vocab.itos = amino_map_rev
    field_tcr.vocab.itos = amino_map_rev
        
    # define dataloader
    if yhat is None:
        fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y)]
        example = [data.Example.fromlist([x1,x2,x3], fields) for x1,x2,x3 in zip(X_pep,X_tcr,y)]
    else:
        fields = [('X_pep',field_pep), ('X_tcr',field_tcr), ('y',field_y), ('yhat',field_yhat)]
        example = [data.Example.fromlist([x1,x2,x3,x4], fields) for x1,x2,x3,x4 in zip(X_pep,X_tcr,y,yhat)]

    dataset = data.Dataset(example, fields)
    loader = data.Iterator(dataset, batch_size=batch_size, device=device0, repeat=False, shuffle=True)

    data_loader = dict()
    data_loader['pep_amino_idx'] = field_pep.vocab.itos
    data_loader['tcr_amino_idx'] = field_tcr.vocab.itos
    data_loader['tensor_type'] = torch.cuda.LongTensor if device == "cuda"  else torch.LongTensor
    data_loader['pep_length'] = maxlen_pep
    data_loader['tcr_length'] = maxlen_tcr 
    data_loader['loader'] = loader

    return data_loader

def tokenizer(sequence):

    sequence = re.sub(r'\s+', '', str(sequence))
    sequence = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX]', '*', sequence)
    sequence = [x for x in sequence]

    return sequence

def load_embedding(filename):

    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum embedding matrix: list 
    '''
    
    f = open(filename, "r")
    lines = f.readlines()[7:]
    f.close()

    embedding = [[float(x) for x in l.strip().split()[1:]] for l in lines]
    embedding.append([0.0] * len(embedding[0]))
    '''    
    blosum = data_io_tf.read_blosum_MN(filename)
    print(blosum)
    amino_map_rev = AMINO_MAP_REV

    embedding = []
    for amino in amino_map_rev:
        embedding.append(blosum[amino])
    '''
    return(embedding)
