import torch
import torch.nn as nn

SIZE_KERNEL1 = 3
SIZE_KERNEL2 = 3

class Net(nn.Module):
    def __init__(self, embedding, pep_length, tcr_length, drop, hiddenDimension, linearSize, blosumFlag):
        
        super(Net, self).__init__()

        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        if blosumFlag:
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze = False)
        
        self.size_hidden1_cnn = 2 * hiddenDimension
        self.size_hidden2_cnn = hiddenDimension
        self.size_kernel1 = SIZE_KERNEL1
        self.size_kernel2 = SIZE_KERNEL2
        self.size_padding = (self.size_kernel1-1)//2

        # Peptide Encoding Layer
        self.encode_pep = nn.Sequential(
            nn.Dropout(drop),
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
            nn.Dropout(drop),
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
        self.size_hidden1_dense = 4 * linearSize
        self.size_hidden2_dense = 2 * linearSize
        self.size_hidden3_dense = 1 * linearSize
        self.net_pep_dim = self.size_hidden2_cnn * ((pep_length-self.size_kernel1+1-self.size_kernel2+1)//self.size_kernel2)
        self.net_tcr_dim = self.size_hidden2_cnn * ((tcr_length-self.size_kernel1+1-self.size_kernel2+1)//self.size_kernel2)
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.net_pep_dim+self.net_tcr_dim, self.size_hidden1_dense),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden2_dense, self.size_hidden3_dense),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden3_dense, 1),
            nn.Sigmoid()
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
