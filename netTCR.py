import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, embedding, args):

        super(Net, self).__init__()

        self.drop_rate = args.drop_rate
        self.n_hid = args.n_hid
        self.n_filters = args.n_filters

        # embedding layer
        self.num_amino = len(embedding)
        self.embedding_dim = len(embedding[0])
        self.embedding = nn.Embedding(self.num_amino,
                                      self.embedding_dim,
                                      padding_idx=self.num_amino-1).\
            from_pretrained(torch.FloatTensor(embedding),
                            freeze=False)

        # peptide encoding layer
        self.pep_encoder1_1 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid())
        self.pep_encoder1_3 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
        self.pep_encoder1_5 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=5, padding=2),
            nn.Sigmoid())
        self.pep_encoder1_7 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=7, padding=3),
            nn.Sigmoid())
        self.pep_encoder1_9 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=9, padding=4),
            nn.Sigmoid())

        # tcr encoding layer
        self.tcr_encoder1_1 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid())
        self.tcr_encoder1_3 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
        self.tcr_encoder1_5 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=5, padding=2),
            nn.Sigmoid())
        self.tcr_encoder1_7 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=7, padding=3),
            nn.Sigmoid())
        self.tcr_encoder1_9 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.n_filters,
                      kernel_size=9, padding=4),
            nn.Sigmoid())

        # peptide-tcr encoding layer
        self.dim1 = args.pep_length + args.tcr_length
        self.encoder2_1 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(self.dim1))
        self.encoder2_3 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(self.dim1))
        self.encoder2_5 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(self.dim1))
        self.encoder2_7 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(self.dim1))
        self.encoder2_9 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters,
                      kernel_size=1, padding=0),
            nn.Sigmoid(),
            nn.MaxPool1d(self.dim1))

        # dense layer at the end
        self.net = nn.Sequential(
            nn.Linear(self.n_filters*5, self.n_hid),
            nn.Sigmoid(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.n_hid, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):

        l_in_pep = self.embedding(pep).transpose(1, 2)
        l_in_tcr = self.embedding(tcr).transpose(1, 2)

        # convolutional layers on peptide
        l_conv_pep_1 = self.pep_encoder1_1(l_in_pep)
        l_conv_pep_3 = self.pep_encoder1_3(l_in_pep)
        l_conv_pep_5 = self.pep_encoder1_5(l_in_pep)
        l_conv_pep_7 = self.pep_encoder1_7(l_in_pep)
        l_conv_pep_9 = self.pep_encoder1_9(l_in_pep)

        # convolutional layers on TCR:
        l_conv_tcr_1 = self.tcr_encoder1_1(l_in_tcr)
        l_conv_tcr_3 = self.tcr_encoder1_3(l_in_tcr)
        l_conv_tcr_5 = self.tcr_encoder1_5(l_in_tcr)
        l_conv_tcr_7 = self.tcr_encoder1_7(l_in_tcr)
        l_conv_tcr_9 = self.tcr_encoder1_9(l_in_tcr)

        l_conc_1 = torch.cat((l_conv_pep_1, l_conv_tcr_1), axis=-1)
        l_conc_3 = torch.cat((l_conv_pep_3, l_conv_tcr_3), axis=-1)
        l_conc_5 = torch.cat((l_conv_pep_5, l_conv_tcr_5), axis=-1)
        l_conc_7 = torch.cat((l_conv_pep_7, l_conv_tcr_7), axis=-1)
        l_conc_9 = torch.cat((l_conv_pep_9, l_conv_tcr_9), axis=-1)

        l_pool_max_1 = self.encoder2_1(l_conc_1).view(-1, self.n_filters)
        l_pool_max_3 = self.encoder2_3(l_conc_3).view(-1, self.n_filters)
        l_pool_max_5 = self.encoder2_5(l_conc_5).view(-1, self.n_filters)
        l_pool_max_7 = self.encoder2_7(l_conc_7).view(-1, self.n_filters)
        l_pool_max_9 = self.encoder2_9(l_conc_9).view(-1, self.n_filters)

        peptcr = torch.cat((l_pool_max_1, l_pool_max_3,
                            l_pool_max_5, l_pool_max_7, l_pool_max_9), -1)
        peptcr = self.net(peptcr)

        return peptcr.view(-1)
