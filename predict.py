import requests
import os
from torch_geometric.data import Dataset
import torch
import pickle
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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


import torch.nn as nn


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=15):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        # h = F.dropout(x, p=0.5, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.val_mask], data.y[data.val_mask].view(-1))
    return acc

dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
print(data.x.shape)
print(range(len(data.x)))
# Create GAT
model = GAT(dataset.num_features, 12, dataset.num_classes)
model = pickle.load(open('gat.pkl', 'rb'))
h, out = model(data.x, data.edge_index)

val_acc = accuracy(out[range(len(data.x))].argmax(dim=1), data.y[range(len(data.x))].view(-1))
print(val_acc)
# Test
#acc = test(model, data)
#print(f'GAT test accuracy: {acc*100:.2f}%\n')

import pandas as pd

def list_to_df(list_values):
  print(list_values)
  df = pd.DataFrame()
  df['idx'] = range(len(list_values))
  df["prediction"] = list_values
  return df


df = list_to_df(out[range(len(data.x))].argmax(dim=1))
df.to_csv('prediction.csv', index=False)
