import torch

import torch.nn as nn
import hiddenlayer as hl


def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 319, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.double()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 319)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


if __name__ == '__main__':
    torch._C.Node.__getitem__ = _node_get

    model = SeismicCNN()
    transforms = [hl.transforms.Prune('Constant')]

    graph = hl.build_graph(model, torch.zeros(1, 1, 129, 2555).double(), transforms=transforms)
    #graph.theme = hl.graph.THEMES['purple'].copy()
    graph.save('./data/lunar/models/SeismicCNN.png', format='png')
