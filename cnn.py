import torch
import torch.nn as nn

# Constants
SIZE_KERNEL1 = 3
SIZE_KERNEL2 = 3

class Net(nn.Module):
    def __init__(self, embedding, args):
        super(Net, self).__init__()

        # Embedding Layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
        if not (args.blosum is None or args.blosum.lower() == 'none'):
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        self.attn_tcr = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)
        self.attn_pep = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)

        """
        # Establish Layer, Kernel, and Padding Sizes
        self.size_hidden1_cnn = 2 * args.n_hid
        self.size_hidden2_cnn = 1 * args.n_hid
        #self.size_hidden3_cnn = args.n_hid
        self.size_kernel1 = SIZE_KERNEL1
        self.size_kernel2 = SIZE_KERNEL2
        self.size_padding = (self.size_kernel1 - 1) // 2

        # Peptide Encoding Layer
        self.encode_pep = nn.Sequential(
            nn.Dropout(args.drop_rate),
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
            #nn.MaxPool1d(kernel_size=self.size_kernel2,
            #             stride=1,
            #             padding=self.size_padding),
            #nn.Conv1d(self.size_hidden2_cnn,
            #          self.size_hidden3_cnn,
            #          kernel_size=self.size_kernel2),
            #nn.BatchNorm1d(self.size_hidden3_cnn),
            #nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
        )

        # TCR Encoding Layer
        self.encode_tcr = nn.Sequential(
            nn.Dropout(args.drop_rate),
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
            #nn.MaxPool1d(kernel_size=self.size_kernel2,
            #             stride=1,
            #             padding=self.size_padding),
            #nn.Conv1d(self.size_hidden2_cnn,
            #          self.size_hidden3_cnn,
            #          kernel_size=self.size_kernel2),
            #nn.BatchNorm1d(self.size_hidden3_cnn),
            #nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=self.size_kernel2)
        )
        """
        # Dense Layer
        self.size_hidden1_dense = 2 * args.lin_size
        self.size_hidden2_dense = 1 * args.lin_size
        #self.size_hidden3_dense = 1 * args.lin_size
        #self.net_pep_dim = self.size_hidden2_cnn * \
        #    ((args.pep_length - self.size_kernel1 + 1 - self.size_kernel2 + 1) // self.size_kernel2)
        #self.net_tcr_dim = self.size_hidden2_cnn * \
        #    ((args.tcr_length - self.size_kernel1 + 1 - self.size_kernel2 + 1) // self.size_kernel2)
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(self.net_pep_dim + self.net_tcr_dim,
                      self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.Dropout(args.drop_rate),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.Dropout(args.drop_rate),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden2_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):

        ## embedding
        pep = self.embedding(pep) # batch * len * dim (25)
        tcr = self.embedding(tcr) # batch * len * dim

        ## attention
        pep, _ = self.attn_pep(pep,pep,pep)
        tcr, _ = self.attn_tcr(tcr,tcr,tcr)

        ## encoder
        #pep = self.encode_pep(pep.transpose(1, 2))
        #tcr = self.encode_tcr(tcr.transpose(1, 2))
        #print(pep.size()) # [32, 43, 25]
        #print(tcr.size()) # [32, 32, 25]

        ## linear
        pep = pep.view(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.view(-1, 1, tcr.size(-2) * tcr.size(-1))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
        peptcr = self.net(peptcr)
        #print(peptcr.size())

        return peptcr
