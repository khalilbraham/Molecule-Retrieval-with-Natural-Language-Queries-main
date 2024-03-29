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

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class GatEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, nheads = 4):
        super(GatEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATConv(num_node_features, graph_hidden_channels,heads=nheads, dropout=0.1, concat=True)
        self.conv2 = GATConv(graph_hidden_channels, graph_hidden_channels,heads=nheads, dropout=0.1,concat=True)
        self.conv3 = GATConv(graph_hidden_channels, graph_hidden_channels,heads=nheads, dropout=0.1,concat=True)
        self.head_linear1 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.head_linear2 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.head_linear3 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.head_linear1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.head_linear2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.head_linear3(x)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class GatEncoderV2(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, nheads = 4):
        super(GatEncoderV2, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels,heads=nheads, dropout=0.1, concat=True)
        self.conv2 = GATv2Conv(graph_hidden_channels, graph_hidden_channels,heads=nheads, dropout=0.1,concat=True)
        self.conv3 = GATv2Conv(graph_hidden_channels, graph_hidden_channels,heads=nheads, dropout=0.1,concat=True)
        self.head_linear1 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.head_linear2 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.head_linear3 = nn.Linear(graph_hidden_channels*nheads, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.head_linear1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.head_linear2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.head_linear3(x)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, graph_encoder, gat_encoder, gatv2_encoder):
        super(EnsembleModel, self).__init__()
        self.graph_encoder = graph_encoder
        self.gat_encoder = gat_encoder
        self.gatv2_encoder = gatv2_encoder

    def forward(self, graph_batch):
        output_graph = self.graph_encoder(graph_batch)
        output_gat = self.gat_encoder(graph_batch)
        output_gatv2 = self.gatv2_encoder(graph_batch)

        # Average the predictions from both models
        ensemble_output = (output_graph + output_gat + output_gatv2) / 3.0

        return ensemble_output
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1024)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return self.linear(encoded_text.last_hidden_state[:, 0, :])

class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, nheads=4):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.gat_encoder = GatEncoder(num_node_features, nout, nhid, graph_hidden_channels, nheads)
        self.gatv2_encoder = GatEncoderV2(num_node_features, nout, nhid, graph_hidden_channels, nheads)
        self.g_encoder =  EnsembleModel(self.graph_encoder, self.gat_encoder, self.gatv2_encoder)
        self.text_encoder = TextEncoder(model_name)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.g_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder