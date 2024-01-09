import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__(aggr='add')  # Use 'add' aggregation function
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x is the node features and edge_index is the adjacency list
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(row, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.linear(aggr_out)

class SimpleGNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GraphConvolution(num_features, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Example usage:
# Assuming you have a graph dataset with features `x` and adjacency list `edge_index`
# and the target labels `y`
num_features = 16
hidden_channels = 32
num_classes = 2

model = SimpleGNN(num_features, hidden_channels, num_classes)

# You should replace the following dummy data with your actual graph data
x = torch.rand((num_nodes, num_features))  # Node features
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Adjacency list
y = torch.tensor([0, 1, 0], dtype=torch.long)  # Target labels

# Forward pass
output = model(x, edge_index)
print(output)
