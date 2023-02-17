import torch
import os
import warnings
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DNN import DNN_R
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore")

def Data_processing(source_location,Traget=True):
    """
    ->Read data in different formats
    ->Data standardization
    ->Scale data
    ->Convert to Tensor format
    ->Data iterator
    """
    if source_location[-4:]=="xlsx":
        all_data=pd.read_excel(source_location).values[:,1:]
    else:
        all_data=pd.read_csv(source_location).values[:,1:]
    if Traget==True:
        Split_ratio=(all_data.shape[0]%50)/all_data.shape[0]
    else:
        if all_data.shape[0]%50<=20:
            Split_ratio=(all_data.shape[0]%50+50)/all_data.shape[0]
            print(f"Source domain data split ratio：{(all_data.shape[0]%50+50)}/{all_data.shape[0]}")
        else:
            Split_ratio=(all_data.shape[0]%50)/all_data.shape[0]
            print(f"Source domain data split ratio：{(all_data.shape[0]%50)}/{all_data.shape[0]}")
    """Data standardization"""
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
    x = np.linspace(0, Epoch,Epoch)
    y=np.array(loss_list)
    plt.plot(x, y, label='Loss Value',marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(fig_name)
    plt.legend()
    plt.savefig(fig_name)
    plt.pause(5)
    plt.close()
def mean_relative_error(y_true, y_pred,):
    """MAPE"""
    return np.average(np.abs((y_true - y_pred) / y_true)*100, axis=0)
def Model_Train(source_location,target_location,net_input=5):
    """
    Training model
    source_ location: source domain file
    target_ location: target domain file
    net_ input: number of neurons in the input layer
    """
    """数据读取"""
    global x_test
    global y_test
    x_train,x_test,y_train,y_test,Train_set,Train_Loader=Data_processing(source_location,Traget=False)
    Tx_train,Tx_test,Ty_train,Ty_test,Target_Train_set,Target_Train_Loader=Data_processing(target_location,Traget=True)
    T_Data=iter(Target_Train_Loader)
    T_iter_num=len(T_Data)
    """Load network model"""
    model = DNN_R(net_input)
    """Initialize network parameters"""
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal(m.weight.data)
            nn.init.xavier_normal(m.weight.data)
            nn.init.kaiming_normal(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    criterion = nn.MSELoss()
    if opt=="SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
        ExpLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    if opt=="Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_list=[]
    min_loss=3000
    save_path='./Output/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.train()
    bar=tqdm(range(max_epoch))
    for epoch in bar:
        all_loss=0
        for j ,data in enumerate(Train_Loader):
            x_train,y_train=data
            t_x_train,t_y_train=T_Data.next()
            if j % T_iter_num == 0:
                T_Data=iter(Target_Train_Loader)
            y_src= model(x_train)
            loss = criterion(y_src, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss+=loss.detach().numpy()
        ExpLR.step()
        bar.set_description("Current Loss: %f" %(all_loss))
        loss_list.append(all_loss)
        if all_loss < min_loss:
            min_loss = all_loss
            save_name = save_path +'best.pth'
            torch.save(model.state_dict(), save_name)
        if epoch==max_epoch-1:
            print("Min_loss=%.4f"%min_loss)
    # plt_loss_curve(loss_list,max_epoch,"Loss Curve")

def Model_Eval(Testpath,savepath,net_input=5):

    if Testpath[-4:]=="xlsx":
        T_Data=pd.read_excel(Testpath)
    else:
        T_Data=pd.read_csv(Testpath)
    all_data=T_Data.values[:,1:]
    target_data=all_data[:,0:-1]
    target_label=all_data[:,-1].reshape(-1,1)

    target_data = StandardScaler().fit_transform(target_data)
    Tx_test = torch.from_numpy(target_data).type(torch.FloatTensor)
    Ty_test = torch.from_numpy(target_label).type(torch.FloatTensor)
    """MAE、MSE"""
    MAE = nn.L1Loss()
    criterion = nn.MSELoss()
    """Model test"""
    model = DNN_R(net_input)
    model_weight_path = './Output/best.pth'
    model.load_state_dict(torch.load(model_weight_path))
    if not os.path.exists("./predict_result/"):
        os.mkdir("./predict_result/")
    model.eval()
    with torch.no_grad():
        # Tx_test=Tx_test.view([-1,net_input,1,1])
        output= model(Tx_test)
        source_output=model(x_test)
        loss = criterion(output, Ty_test)
        """The prediction results are written in the csv table"""
        data_result = {'real_value': np.around(Ty_test.numpy().flatten(), 2).tolist(),
                       'pred_value': np.around(output.numpy().flatten(), 2).tolist()}
        dataframe = pd.DataFrame(data_result, columns=['real_value', 'pred_value'])
        concat_result = pd.concat([T_Data, dataframe], axis=1)
        concat_result.to_csv("./predict_result/"+savepath, index=False)
        predict_list = output.detach().numpy()
        MAE_LOSS=MAE(output,Ty_test)
        print("||*******************************************************************||")
        print('Average absolute error of target domain:%.4f'%MAE_LOSS.detach().numpy())
        print('Mean square error of target domain:%.4f'%loss.detach().numpy())
        print(f"Mean absolute percentage error of target domain：{np.around(mean_relative_error(Ty_test.detach().numpy(),predict_list),4)}")
        print("||*******************************************************************||")

    return MAE_LOSS.detach().numpy(),loss.detach().numpy(),np.around(mean_relative_error(Ty_test.detach().numpy(),predict_list),4)

def main(seed,txt):
    torch.manual_seed(seed)
    Model_Train(source_location,target_location)
    Mae,Mse,Re=Model_Eval(Test_path,save_path)
    params = ['seed:'+str(seed),
              "epoch："+str(max_epoch),
         'source_location:'+source_location,
         'target_location:'+target_location,
         'save_path:'+save_path,
         'Test_path:'+Test_path,
         'optimizer:'+opt,
         'MAE:'+str(Mae),
         "MSE:"+str(Mse),
         "MAPE:"+str(Re)]
    if not os.path.exists("./predict_result/"+txt+'/'):
        os.mkdir("./predict_result/"+txt+"/")
    f = open("./predict_result/"+txt+"/"+str(txt)+'.txt','w')
    for param in params:
        f.writelines(param)
        f.write('\n')
    f.close()
if __name__=="__main__":
    # Epochs
    max_epoch = 100
    # Get the absolute path of the current file
    currentPath = os.getcwd().replace('\\','/')
    # Source domain file path
    source_location = currentPath + '/Source_data/por/L1-L2-L3.xlsx'
    # Target domain core file path
    target_location = currentPath + '/target_data/por/A2-POR.csv'
    save_path="A2.csv"
    Test_path= target_location
    opt="Adam"
    main(seed=5,txt='A2'+opt)

