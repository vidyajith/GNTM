#!/usr/bin/env python
# coding: utf-8

# # How to explain Graph Neural Networks using GNNExplainer

# ## Install Pytorch Geometric

# In[ ]:


import torch
torch.manual_seed(42)
from IPython.display import clear_output 
torch_version = torch.__version__
print("Torch version: ", torch_version)
pytorch_version = f"torch-{torch.__version__}.html"
#get_ipython().system('pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/$pytorch_version')
#get_ipython().system('pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/$pytorch_version')
#get_ipython().system('pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/$pytorch_version')
#get_ipython().system('pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/$pytorch_version')
#get_ipython().system('pip install torch-geometric')
#clear_output()
print("Done.")


# ## Twitch Streamer Dataset
# 
# - Twitch user-user networks of gamers who stream in a certain language
# - Nodes are the users themselves and the links are mutual friendships between them
# - These social networks were collected in May 2018

# In[ ]:


from torch_geometric.datasets import Twitch
# Dataset source: https://github.com/benedekrozemberczki/datasets#twitch-social-networks
graph = Twitch(root=".", name="EN")[0]
graph


# ### The node features 
# - Extracted based on the games played and liked, location and streaming habits
# - These are embeddings, which cannot be interpreted directly

# In[ ]:


graph.x


# ### A binary node classification task
# - Predict if a streamer uses explicit language

# In[ ]:


graph.y


# In[ ]:


import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 5, 5
df = pd.DataFrame(graph.y.numpy(), columns=["explicit_language"])
df['explicit_language'].value_counts().plot(kind='bar')


# ## A simple model

# In[ ]:


from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GATConv
embedding_size = 128

class GNN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GNN, self).__init__()

        # GCN layers
        self.initial_conv = GATConv(graph.num_features, embedding_size)
        self.conv1 = GATConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size, 1)

    def forward(self, x, edge_index):
        emb = F.relu(self.initial_conv(x, edge_index))
        emb = F.relu(self.conv1(emb, edge_index))
        return self.out(emb)

model = GNN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


# ### Add train and test masks
# - Alternative: Transform + RandomNodeSplit

# In[ ]:


# Add train and test masks
num_nodes = graph.x.shape[0]
ones = torch.ones(num_nodes)
ones[4000:] = 0
graph.train_mask = ones.bool()
graph.test_mask = ~graph.train_mask.bool()

print("Train nodes: ", sum(graph.train_mask))
print("Test nodes: ", sum(graph.test_mask))


# In[ ]:


graph.train_mask


# In[ ]:


graph.test_mask


# ## Training
# 
# - We train the model with the full graph, so batch size = 1
# - For large graphs its also possible to train in batches, by using the NeighborSampler
# - This is a transductive setup, as all nodes are used during training
# - Inductive training can be achieved using GraphSAGE
# 

# In[ ]:


# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GNN()
model = model.to(device)
graph = graph.to(device)

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  


def train():
      model.train()
      optimizer.zero_grad() 
      out = model(graph.x, graph.edge_index)  
      preds = out[graph.train_mask]
      targets = torch.unsqueeze(graph.y[graph.train_mask], 1)
      loss = loss_fn(preds.float(), targets.float())  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      optimizer.zero_grad() 
      out = model(graph.x, graph.edge_index)  
      preds = out[graph.test_mask]
      targets = torch.unsqueeze(graph.y[graph.test_mask], 1)
      loss = loss_fn(preds.float(), targets.float())  
      return loss

for epoch in range(0, 800):
    tr_loss = train()
    if epoch % 100 == 0:
      loss = test()
      print(f'Epoch: {epoch:03d}, Test loss: {loss:.4f} | Train loss: {tr_loss:.4f}')


# In[ ]:


from sklearn.metrics import roc_auc_score

df = pd.DataFrame()
# Model predictions'
out = torch.sigmoid(model(graph.x, graph.edge_index))
df["preds"] = out[graph.test_mask].round().int().cpu().detach().numpy().squeeze()
df["prob"] = out[graph.test_mask].cpu().detach().numpy().squeeze().round(2)

# Groundtruth
df["gt"] = graph.y[graph.test_mask].cpu().detach().numpy().squeeze()

print("Test ROC: ", roc_auc_score(df["gt"], df["preds"]))
df.head(10)


# ## Explaining the predictions

# In[ ]:


from torch_geometric.nn import GNNExplainer
# Initialize explainer
explainer = GNNExplainer(model, epochs=200, return_type='log_prob')

# Explain node
node_idx = 7
node_feat_mask, edge_mask = explainer.explain_node(node_idx, graph.x, graph.edge_index)
print("Size of explanation: ", sum(edge_mask > 0))


# In[ ]:


print("Node features: ", graph.x[node_idx])
print("Node label: ",  df["gt"][node_idx])
print("Node prediction: ",  df["preds"][node_idx])


# ## Create visualizations

# In[ ]:


# Show shape of masks
print(node_feat_mask.shape)
print(edge_mask.shape)


# ### The size of the following plot depends on the depth of the GNN!
# - Colors are edge labels
# - Greyed-out edges have an impact, but very low (< 0.1)

# In[ ]:


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

# Visualize result
ax, G = explainer.visualize_subgraph(node_idx, graph.edge_index, edge_mask, y=graph.y)
plt.show()


# #### For further details: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
