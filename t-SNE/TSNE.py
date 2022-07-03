import torch
import mmd
import os
import warnings
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from vit_model import vit_base_patch16_224_in21k
from DaNN import DaNN_R
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE
from TSNE_fuc import tsne
def Data_processing(source_location,Traget=True):

    if source_location[-4:]=="xlsx":
        all_data=pd.read_excel(source_location).values[:,1:]
    else:
        all_data=pd.read_csv(source_location).values[:,1:]
    """Adaptive data segmentation scale"""
    if Traget==True:
        Split_ratio=(all_data.shape[0]%50)/all_data.shape[0]
        print(f"Target domain data segmentation ratio:{(all_data.shape[0]%50)}/{all_data.shape[0]}")
    else:
        if all_data.shape[0]%50<=10:
            Split_ratio=(all_data.shape[0]%50+50)/all_data.shape[0]
            print(f"Source domain data segmentation ratio：{(all_data.shape[0]%50+50)}/{all_data.shape[0]}")
        else:
            Split_ratio=(all_data.shape[0]%50)/all_data.shape[0]
            print(f"Source domain data segmentation ratio：{(all_data.shape[0]%50)}/{all_data.shape[0]}")
    x=StandardScaler().fit_transform(all_data[:,0:-1])
    y=all_data[:,-1].reshape(-1,1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=Split_ratio,random_state=55)
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    Train_set = TensorDataset(x_train,y_train)
    Train_Loader = DataLoader(dataset=Train_set,batch_size=50,shuffle=True)
    return  x_train,x_test,y_train,y_test,Train_set,Train_Loader
def mmd_loss(x_src, x_tar,GAMMA = 10 ^ 3):
    """mmd"""
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])
def draw_infer_result(groud_truths,infer_results,title):
    plt.title(title, fontsize=24)
    x= np.arange(0,10)
    plt.plot(x, x)
    plt.xlabel('Ground Truth', fontsize=14)
    plt.ylabel('Infer Result', fontsize=14)
    plt.scatter(groud_truths, infer_results,color='green')
    plt.grid()
    plt.savefig(title)
    plt.show()
def plt_loss_curve(loss_list,Epoch,fig_name):
    plt.clf()
    x = np.linspace(0, Epoch,Epoch)
    y=np.array(loss_list)[0:500]
    plt.plot(x, y, label=fig_name,color="red")
    plt.xlabel('Iterations',fontsize=13,weight='bold')
    plt.ylabel('Loss',fontsize=13,weight='bold')
    plt.savefig("LOSSIMAG/"+fig_name)
    plt.savefig("LOSSIMAG/"+fig_name + '.svg', format='svg', bbox_inches='tight')


def mean_relative_error(y_true, y_pred,):
    return np.average(np.abs((y_true - y_pred) / y_true)*100, axis=0)
def Model_Train(source_location,target_location,net_input=5):
    x_train,x_test,y_train,y_test,Train_set,Train_Loader=Data_processing(source_location,Traget=False)
    Tx_train,Tx_test,Ty_train,Ty_test,Target_Train_set,Target_Train_Loader=Data_processing(target_location,Traget=True)
    T_Data=iter(Target_Train_Loader)
    T_iter_num=len(T_Data)
    model = DaNN_R(net_input)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_list=[]
    min_loss=3000
    save_path='./Output/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists("./tsnePlot/"):
        os.mkdir("./tsnePlot/")
    if not os.path.exists("./LOSSIMAG/"):
        os.mkdir("./LOSSIMAG/")

    bar=tqdm(range(max_epoch))
    for epoch in bar:
        "Defining characteristic variables"
        Src_Feature = []
        Tar_Feature = []

        model.train()
        all_loss=0
        for j ,data in enumerate(Train_Loader):
            x_train,y_train=data
            t_x_train,t_y_train=T_Data.next()
            if j % T_iter_num == 0:
                T_Data=iter(Target_Train_Loader)

            y_src, x_src_mmd, x_tar_mmd,x_src_64,x_tar_64 ,x_src_31,x_tar_31= model(x_train,t_x_train)
            Src_Feature.append(x_src_mmd.detach().numpy())
            Tar_Feature.append(x_tar_mmd.detach().numpy())

            loss = criterion(y_src, y_train)
            loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)  # MMD对齐
            loss_mmd_64 = mmd_loss(x_src_64, x_tar_64)
            loss_mmd_31 = mmd_loss(x_src_31, x_tar_31)

            loss = loss_mmd + loss + loss_mmd_64 + loss_mmd_31
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss+=loss.detach().numpy()
            loss_list.append(loss.detach().numpy())
        """Characteristic data drawing"""
        tsne(Src_Feature,Tar_Feature,epoch)
        ExpLR.step()
        bar.set_description("Current Loss: %f" %(all_loss))
        if all_loss < min_loss:
            min_loss = all_loss
            save_name = save_path +'best.pth'
            torch.save(model.state_dict(), save_name)
        if epoch==max_epoch-1:
            print("Min_loss=%.4f"%min_loss)

    plt_loss_curve(loss_list,500,"POR loss curve of well A1 ")


if __name__=="__main__":
    torch.manual_seed(22)
    max_epoch = 100
    currentPath = os.getcwd().replace('\\','/')
    source_location = currentPath + '/data/N1-N2-N3.xlsx'
    target_location = currentPath + '/data/A1_all_data.csv'
    Test_path=target_location
    Model_Train(source_location,target_location)


