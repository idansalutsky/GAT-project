import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torch_geometric.nn import GATv2Conv
import torch
from torch_geometric.nn import BatchNorm
import pickle
import requests
import os
from torch_geometric.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
data = dataset[0].to(device)


class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=10, dropout=0.08)
        self.bn1 = BatchNorm(10 * dim_h)
        self.gat2 = GATv2Conv(10 * dim_h, dim_out, heads=1, concat=False)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.prelu = torch.nn.PReLU()


    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.gat2(x, edge_index)
        return x, self.logsoftmax(x)


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()



def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 450
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
            torch.save(model.state_dict(), 'gat_model.pt')


    return model


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.val_mask], data.y[data.val_mask].view(-1))
    return acc



# Create GAT
gat = GAT(dataset.num_features, 48, dataset.num_classes).to(device)
print(gat)

# Train
train(gat, data)

# Test
acc = test(gat, data)
print(f'GAT test accuracy: {acc*100:.2f}%\n')







