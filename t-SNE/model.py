import torch.nn as nn
import torch
import torch.nn.functional as F
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class DaNN_R(nn.Module):
    def __init__(self, n_input=5):
        super(DaNN_R, self).__init__()
        self.layer_input1 = nn.Linear(n_input, 31)
        self.layer_input2 = nn.Linear(31, 64)
        self.layer_input3 = nn.Linear(64, 31)
        self.layer_input4 = nn.Linear(31, 1)
        self.layer_input5=nn.Linear(31,31)
        self.relu = nn.GELU()
        self.layer_input6=nn.Identity()
        # self.relu = Mish()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, src, tar):

        x_src = self.dropout(self.relu(self.layer_input1(src)))
        x_tar = self.dropout(self.relu(self.layer_input1(tar)))

        x_src = self.dropout(self.relu(self.layer_input5(x_src)))
        x_tar = self.dropout(self.relu(self.layer_input5(x_tar)))
        Res_x_src = x_src
        Res_x_tar = x_tar

        x_src = self.dropout(self.relu(self.layer_input2(x_src)))
        x_tar = self.dropout(self.relu(self.layer_input2(x_tar)))

        x_src_64 = x_src
        x_tar_64 = x_tar

        x_src = self.dropout(self.relu(self.layer_input3(x_src)))
        x_tar = self.dropout(self.relu(self.layer_input3(x_tar)))

        x_src_mmd = x_src
        x_src_mmd=self.layer_input6(x_src_mmd)
        x_tar_mmd = x_tar
        y_src = self.layer_input4(x_src)
        return y_src, x_src_mmd, x_tar_mmd,x_src_64,x_tar_64,Res_x_src,Res_x_tar

