import numpy as np
import requests
import os
from torch_geometric.data import Dataset
import torch
import pickle
from torch_geometric.nn import GATv2Conv, BatchNorm


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

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=8, dropout=0.09)
        self.bn1 = BatchNorm(8 * dim_h)
        self.gat2 = GATv2Conv(8 * dim_h, dim_h, heads=7, dropout=0.09)
        self.bn2 = BatchNorm(7 * dim_h)
        self.gat3 = GATv2Conv(7 * dim_h, dim_h, heads=7, dropout=0.09)
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
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def testalldata(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[range(len(data.x))], data.y[range(len(data.x))].view(-1))
    return acc

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.val_mask], data.y[data.val_mask].view(-1))
    return acc

dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
print("========================")

model = GAT(dataset.num_features, 48, dataset.num_classes)
model = pickle.load(open('gat_model.pkl', 'rb'))
h, out = model(data.x, data.edge_index)

#val_acc = test(model, data)
#print(val_acc)
#val_acc = testalldata(model, data)
#print(val_acc)


import pandas as pd

def list_to_df(list_values):
  print(list_values)
  df = pd.DataFrame()
  df['idx'] = range(len(list_values))
  df["prediction"] = list_values
  return df


df = list_to_df(out[range(len(data.x))].argmax(dim=1))
df.to_csv('prediction.csv', index=False)

