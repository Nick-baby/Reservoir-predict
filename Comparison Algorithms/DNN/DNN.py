import torch.nn as nn

class DaNN(nn.Module):
    def __init__(self, n_input=28 * 28, n_hidden=256, n_class=10):
        super(DaNN, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(n_hidden, n_class)

    def forward(self, src, tar):
        x_src = self.layer_input(src)
        x_tar = self.layer_input(tar)
        x_src = self.dropout(x_src)
        x_tar = self.dropout(x_tar)
        x_src_mmd = self.relu(x_src)
        x_tar_mmd = self.relu(x_tar)
        y_src = self.layer_hidden(x_src_mmd)
        return y_src, x_src_mmd, x_tar_mmd

class DNN_R(nn.Module):
    def __init__(self, n_input=5):
        super(DNN_R, self).__init__()
        self.layer_input1 = nn.Linear(n_input, 31)
        self.layer_input5=nn.Linear(31,31)
        self.layer_input2 = nn.Linear(31, 64)

        self.layer_input3 = nn.Linear(64, 31)
        self.layer_input4 = nn.Linear(31, 1)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=0)
    def forward(self, src):
        x_src= self.dropout(self.relu(self.layer_input1(src)))
        x_src = self.dropout(self.relu(self.layer_input5(x_src)))
        x_src = self.dropout(self.relu(self.layer_input2(x_src)))
        x_src = self.dropout(self.relu(self.layer_input3(x_src)))
        y_src = self.layer_input4(x_src)
        # x_src = self.dropout(self.relu(self.layer_input1(src)))
        # x_src = self.relu(self.layer_input5(x_src))
        # x_src = self.relu(self.layer_input2(x_src))
        # x_src = self.relu(self.layer_input3(x_src))
        # y_src = self.layer_input4(x_src)
        return y_src

