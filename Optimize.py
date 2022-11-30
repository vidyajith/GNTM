

import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from graphlime import GraphLIME
import matplotlib.pyplot as plt

#%matplotlib inline

dataset = Planetoid('.', 'Cora', transform=NormalizeFeatures())
data = dataset[0]
data

plt.figure(figsize=(16, 12))

G = to_networkx(data, to_undirected=True)
pos = nx.spring_layout(G, k=0.1)
nx.draw_networkx(G, with_labels=False, node_size=30)

plt.title('Cora');
plt.show()

#extract the edges

degrees = dict(nx.degree(G))
node_indices = list(degrees.keys())
node_degrees = list(degrees.values())

max_degree = max(node_degrees)
node_idx = np.argmax(node_degrees).item()


class GAT(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 heads_1=8, heads_2=1, att_dropout=0.6, input_dropout=0.6):
        super(GAT, self).__init__()

        self.att_dropout = att_dropout
        self.input_dropout = input_dropout

        self.conv1 = GATConv(in_channels=input_dim,
                             out_channels=hidden_dim // heads_1,
                             heads=heads_1,
                             concat=True,
                             dropout=att_dropout)
        self.conv2 = GATConv(in_channels=hidden_dim,
                             out_channels=output_dim,
                             heads=heads_2,
                             concat=False,
                             dropout=att_dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


hparams = {
    'input_dim': data.num_node_features,
    'hidden_dim': 16,
    'output_dim': max(data.y).item() + 1
}

model = GAT(**hparams)
model


def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


lr = 0.005
epochs = 400

model.train()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()

    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = accuracy(output[data.train_mask], data.y[data.train_mask])
        print('Epoch: {:3d}, acc = {:.3f}'.format(epoch, acc))

print(node_idx)

print(model.eval())

# instantiate a GraphLIME object
explainer = GraphLIME(model, hop=2, rho=0.1, cached=True)

# explain node features by calling the method `explain_node()`
coefs = explainer.explain_node(node_idx, data.x, data.edge_index)

print(coefs)

plt.figure(figsize=(16, 4))

x = list(range(data.num_node_features))

plt.bar(x, coefs, width=5.0)
plt.xlabel('Feature Index')
plt.ylabel(r'$\beta$');
plt.show()

print(f'The {np.argmax(coefs)}-th feature is the most important.')