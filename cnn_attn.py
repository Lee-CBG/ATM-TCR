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

        self.attn_tcr = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads, dropout=0.1)
        self.attn_tcr2 = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads, dropout=0.1)
        self.attn_pep = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads, dropout=0.1)
        self.attn_pep2 = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads, dropout=0.1)
        self.layer_norm_tcr = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_tcr2 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_pep = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_pep2 = nn.LayerNorm(self.embedding_dim)

        # Dense Layer
        self.size_hidden1_dense = 4 * args.lin_size
        self.size_hidden2_dense = 2 * args.lin_size
        self.size_hidden3_dense = 1 * args.lin_size
        self.net_pep_dim = args.max_len_pep * self.embedding_dim
        self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
        self.net = nn.Sequential(
            nn.Dropout(args.drop_rate),
            nn.Linear(self.net_pep_dim + self.net_tcr_dim, self.size_hidden1_dense),
            nn.BatchNorm1d(self.size_hidden1_dense),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
            nn.BatchNorm1d(self.size_hidden2_dense),
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden2_dense, self.size_hidden3_dense),
            nn.BatchNorm1d(self.size_hidden3_dense),            
            nn.LeakyReLU(),
            nn.Linear(self.size_hidden3_dense, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):

        # Embedding
        pep = self.embedding(pep) # batch * len * dim (25)
        tcr = self.embedding(tcr) # batch * len * dim

        # Attention
        pep, _ = self.attn_pep(pep,pep,pep)
        pep = self.layer_norm_pep(pep)
        pep, _ = self.attn_pep2(pep,pep,pep)
        pep = self.layer_norm_pep2(pep)
        tcr, _ = self.attn_tcr(tcr,tcr,tcr)
        tcr = self.layer_norm_tcr(tcr)
        tcr, _ = self.attn_tcr2(tcr,tcr,tcr)
        tcr = self.layer_norm_tcr2(tcr)

        # Dense Layer
        pep = pep.view(-1, 1, pep.size(-2) * pep.size(-1))
        tcr = tcr.view(-1, 1, tcr.size(-2) * tcr.size(-1))
        peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
        peptcr = self.net(peptcr)

        return peptcr
