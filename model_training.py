import pickle
import requests
import os
from torch_geometric.data import Dataset



class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])



dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)


class GAT(nn.Module):
    def __init__(self, n_features, n_classes, n_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha


        self.attentions = [GraphAttentionLayer(n_features, n_classes, dropout, alpha) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_features * n_heads, n_classes, dropout, alpha)

    def forward(self, input, adj):
        x = F.dropout(input, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out, heads=15):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.5, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

"""
import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=8, dropout=0.08)
        self.bn1 = BatchNorm(8 * dim_h)
        self.gat2 = GATv2Conv(8 * dim_h, dim_h, heads=7, dropout=0.08)
        self.bn2 = BatchNorm(7 * dim_h)
        self.gat3 = GATv2Conv(7 * dim_h, dim_h, heads=7, dropout=0.08)
        self.bn4 = BatchNorm(7 * dim_h)
        self.gat4 = GATv2Conv(7 * dim_h, dim_out, heads=1, concat=False)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.prelu = torch.nn.PReLU()


    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = self.prelu(x)
        x = self.gat3(x, edge_index)
        x = self.bn4(x)
        x = self.prelu(x)
        x = self.gat4(x, edge_index)
        return x, self.logsoftmax(x)

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()



def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 350
    best_acc = 0

    model.train()
    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].view(-1))
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask].view(-1))
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask].view(-1))
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask].view(-1))

        # Print metrics every 10 epochs
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc * 100:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            pickle.dump(model, open('gat_model.pkl', 'wb'))

    return model


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.val_mask], data.y[data.val_mask].view(-1))
    return acc



# Create GAT
gat = GAT(dataset.num_features, 48, dataset.num_classes)
print(gat)

# Train
train(gat, data)

# Test
acc = test(gat, data)
print(f'GAT test accuracy: {acc*100:.2f}%\n')


