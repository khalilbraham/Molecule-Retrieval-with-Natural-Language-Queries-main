from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv as GATConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

from torch import nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

class GATEncoderAdv(nn.Module):
    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoderAdv, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, hidden_channels = self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden, hidden_channels = self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout)
        self.gatenc3 = GATv2Conv(in_channels=self.n_hidden, hidden_channels = self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout)
        self.gatenc4 = GATv2Conv(in_channels=self.n_hidden, hidden_channels = self.attention_hidden, out_channels=self.n_hidden, dropout=self.dropout)

        #self.self_attn = nn.MultiheadAttention(self.n_hidden, num_heads=16, dropout=self.dropout)
        self.res_conn = nn.ModuleList()
        for _ in range(3):
            self.res_conn.append(nn.Linear(self.n_hidden, self.n_hidden))
            self.res_conn.append(nn.ReLU())

        self.res_conn.append(nn.Linear(self.n_hidden, self.attention_hidden))
        self.res_conn.append(nn.ReLU()) 

        self.mol_hidden1 = nn.Linear(self.attention_hidden, nhid*2)
        self.mol_hidden2 = nn.Linear(nhid*2, nout)
        self.ln = nn.LayerNorm((nout))


    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

       
        x = self.gatenc1(x, edge_index)
        x = x.relu()
        x = self.res_conn[0](x) + x
        x = self.gatenc2(x, edge_index)
        x = x.relu()
        x = self.res_conn[1](x) + x
        x = self.gatenc3(x, edge_index)
        x = x.relu()
        x = self.res_conn[2](x) + x
        x = self.gatenc4(x, edge_index)
        x = x.relu()
        x = self.res_conn[3](x) + x

        # aggregate node representations
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        x = self.ln(x)
        return x

    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1024)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return self.linear(encoded_text.last_hidden_state[:, 0, :])

class Model(nn.Module):
    def __init__(self, model_name, nout, nhid, attention_hidden, n_in, dropout):
        super(Model, self).__init__()
        self.graph_encoder = GATEncoderAdv(nout, nhid, attention_hidden, n_in, dropout)
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder