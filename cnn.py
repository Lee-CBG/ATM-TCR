import torch
import torch.nn as nn

SIZE_HIDDEN1_CNN = 64
SIZE_HIDDEN2_CNN = 32

SIZE_KERNEL1 = 3
SIZE_KERNEL2 = 3

class Net(nn.Module):
    def __init__(self, embedding, pep_length, tcr_length):
        
        super(Net, self).__init__()

        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino,
                                      self.embedding_dim,
                                      padding_idx=self.num_amino-1).\
                                      from_pretrained(torch.FloatTensor(embedding),
                                                      freeze = False)

        
        
        self.size_hidden1_cnn = SIZE_HIDDEN1_CNN
        self.size_hidden2_cnn = SIZE_HIDDEN2_CNN
        self.size_kernel1 = SIZE_KERNEL1
        self.size_kernel2 = SIZE_KERNEL2
        self.size_padding = (self.size_kernel1-1)//2

        # Peptide Encoding Layer
        self.encode_pep = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(self.embedding_dim,
                      self.size_hidden1_cnn,
                      kernel_size=self.size_kernel1),
            nn.BatchNorm1d(self.size_hidden1_cnn),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel1,
                         stride=1,
                         padding=self.size_padding),
            nn.Conv1d(self.size_hidden1_cnn,
                      self.size_hidden2_cnn,
                      kernel_size=self.size_kernel2),
            nn.BatchNorm1d(self.size_hidden2_cnn),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
            )
        
        # TCR Encoding Layer
        self.encode_tcr = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(self.embedding_dim,
                      self.size_hidden1_cnn,
                      kernel_size=self.size_kernel1),
            nn.BatchNorm1d(self.size_hidden1_cnn),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel1,
                         stride=1,
                         padding=self.size_padding),
            nn.Conv1d(self.size_hidden1_cnn,
                      self.size_hidden2_cnn,
                      kernel_size=self.size_kernel2),
            nn.BatchNorm1d(self.size_hidden2_cnn),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
            )

        # Dense Layer
        self.net_pep_dim = self.size_hidden2_cnn * ((pep_length-self.size_kernel1+1-self.size_kernel2+1)//self.size_kernel2)
        self.net_tcr_dim = self.size_hidden2_cnn * ((tcr_length-self.size_kernel1+1-self.size_kernel2+1)//self.size_kernel2)
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.LogSoftmax(1)
            )

    def forward(self, pep, tcr):

        pep = self.embedding(pep)
        tcr = self.embedding(tcr)
        pep = self.encode_pep(pep.transpose(1,2))
        tcr = self.encode_tcr(tcr.transpose(1,2))
        peptcr = torch.cat((pep, tcr), -1)
        peptcr = peptcr.view(-1, 1, peptcr.size(-1) * peptcr.size(-2)).squeeze(-2)
        peptcr = self.net(peptcr)
        
        return peptcr