import torch.nn as nn

class ModelHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        '''
        A head on top of VGG16 model.
        A classifier model to detect cat vs dog.

        Args:
            input_size: input size of the data

        '''
        super(ModelHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sig(out)

        return out