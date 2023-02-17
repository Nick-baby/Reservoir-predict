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
from model import DaNN_R
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore")
def Data_processing(source_location,Traget=True):

    if source_location[-4:]=="xlsx":
        all_data=pd.read_excel(source_location).values[:,1:]
    else:
        all_data=pd.read_csv(source_location).values[:,1:]
    """Adaptive data split ratio"""
    if Traget==True:
        Split_ratio=(all_data.shape[0]%50)/all_data.shape[0]
        print(f"Target domain data split ratio:{(all_data.shape[0]%50)}/{all_data.shape[0]}")
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
def mmd_loss(x_src, x_tar,GAMMA = 10 ^ 3):
    """mmd"""
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])
def draw_infer_result(groud_truths,infer_results,title):
    '''Plot the actual and predicted values'''
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
    """Symmetric training loss graph"""
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
def plt_loss_curve_iter(loss_list,Epoch,fig_name):
    """Training loss graph"""
    # plt.clf()  # 清图
    x = np.linspace(0, Epoch,Epoch)
    y=np.array(loss_list)[0:800]
    plt.plot(x, y, label=fig_name,color="red")
    plt.xlabel('Iterations',fontsize=13,weight='bold')
    plt.ylabel('Loss',fontsize=13,weight='bold')
    # plt.title(fig_name)
    # plt.legend()
    plt.savefig(fig_name)
    # plt.savefig("LOSSIMAG/"+fig_name + '.svg', format='svg', bbox_inches='tight')

    plt.pause(5)
    plt.close()

def mean_relative_error(y_true, y_pred,):
    """MAPE"""
    return np.average(np.abs((y_true - y_pred) / y_true)*100, axis=0)
def Model_Train(source_location,target_location,net_input=5):

    global x_test
    global y_test
    x_train,x_test,y_train,y_test,Train_set,Train_Loader=Data_processing(source_location,Traget=False)
    Tx_train,Tx_test,Ty_train,Ty_test,Target_Train_set,Target_Train_Loader=Data_processing(target_location,Traget=True)
    T_Data=iter(Target_Train_Loader)
    T_iter_num=len(T_Data)
    model = DaNN_R(net_input)
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal(m.weight.data)
            nn.init.xavier_normal(m.weight.data)
            nn.init.kaiming_normal(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)
    criterion = nn.MSELoss()
    """Optimizer settings - SGD or Adam"""
    if opt=="SGD":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
        ExpLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    if opt=="Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    loss_list=[]
    t_list=[]
    loss_mmd_list=[]
    min_loss=3000
    save_path='./Output/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model.train()
    bar=tqdm(range(max_epoch))
    for epoch in bar:
        all_loss=0
        loss_mmd_all=0
        for j ,data in enumerate(Train_Loader):
            x_train,y_train=data
            t_x_train,t_y_train=T_Data.next()
            if j % T_iter_num == 0:
                T_Data=iter(Target_Train_Loader)
            y_src, x_src_mmd, x_tar_mmd,x_src_64,x_tar_64 ,x_src_31,x_tar_31= model(x_train,t_x_train) 

            loss = criterion(y_src, y_train)
            loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
            loss_mmd_64 = mmd_loss(x_src_64, x_tar_64)
            loss_mmd_31 = mmd_loss(x_src_31, x_tar_31)
            loss =  loss_mmd + loss + loss_mmd_64 + loss_mmd_31
            loss_MMD = loss_mmd  + loss_mmd_64 + loss_mmd_31
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss+=loss.detach().numpy()
            loss_mmd_all+=loss_MMD
            t_list.append(loss.detach().numpy())

        ExpLR.step()
        bar.set_description("Current Loss: %f" %(all_loss))
        loss_list.append(all_loss)
        loss_mmd_list.append(loss_mmd_all.detach().numpy())
        """Save the minimum loss value of the model"""
        if all_loss < min_loss:
            min_loss = all_loss
            save_name = save_path +'best.pth'
            torch.save(model.state_dict(), save_name)
        if epoch==max_epoch-1:
            print("Save the minimum loss value of the model：Min_loss=%.4f"%min_loss)
    """Plot the training loss curve, save and display for 5 seconds"""
    # plt_loss_curve(loss_list,max_epoch,"Loss Curve")
    plt_loss_curve_iter(t_list,800,"POR loss curve of well A2 ")
def Model_Eval(Testpath,savepath,net_input=5):
    """
    Validate the model
    Testpath：test file
    savepath：Prediction result save file
    net_input：The number of neurons in the input layer
    """
    """Read different formats of data files"""
    if Testpath[-4:]=="xlsx":
        T_Data=pd.read_excel(Testpath)
    else:
        T_Data=pd.read_csv(Testpath)
    """Load test data"""
    all_data=T_Data.values[:,1:]
    target_data=all_data[:,0:-1]
    target_label=all_data[:,-1].reshape(-1,1)
    print(f"Target data number：{target_data.shape[0]}")
    """Target domain data normalization -> convert to Tensor"""
    target_data = StandardScaler().fit_transform(target_data)
    Tx_test = torch.from_numpy(target_data).type(torch.FloatTensor)
    Ty_test = torch.from_numpy(target_label).type(torch.FloatTensor)
    """Define evaluation indicators MAE, MSE"""
    MAE = nn.L1Loss()
    criterion = nn.MSELoss()
    """test -- load the best model for testing"""
    model = DaNN_R(net_input)
    model_weight_path = './Output/best.pth'
    model.load_state_dict(torch.load(model_weight_path))
    """Create prediction result save path"""
    if not os.path.exists("./predict_result/"):
        os.mkdir("./predict_result/")
    """assessment test"""
    model.eval()
    with torch.no_grad():
        output, _, _ ,_,_,_,_= model(Tx_test,Tx_test)
        source_output, _, _ ,_,_,_,_=model(x_test,x_test)
        loss = criterion(output, Ty_test)
        """Prediction results are written to a csv table"""
        data_result = {'real_value': np.around(Ty_test.numpy().flatten(), 2).tolist(),
                       'pred_value': np.around(output.numpy().flatten(), 2).tolist()}
        dataframe = pd.DataFrame(data_result, columns=['real_value', 'pred_value'])
        concat_result = pd.concat([T_Data, dataframe], axis=1)
        concat_result.to_csv("./predict_result/"+savepath, index=False)
        predict_list = output.detach().numpy()
        MAE_LOSS=MAE(output,Ty_test)
        print("||*******************************************************************||")
        print('target MAE:%.4f'%MAE_LOSS.detach().numpy())
        print('Target MSE:%.4f'%loss.detach().numpy())
        print(f"Target MAPE：{np.around(mean_relative_error(Ty_test.detach().numpy(),predict_list),4)}")
        print("||*******************************************************************||")
        """Prediction Results Visualization"""
        # draw_infer_result(Ty_test.detach().numpy(),predict_list,'A1 result')
    return MAE_LOSS.detach().numpy(),loss.detach().numpy(),np.around(mean_relative_error(Ty_test.detach().numpy(),predict_list),4)
def serch_best_seed(scale):
    mae_record=[]
    mse_record=[]
    re_record=[]
    min_mse_loss=1000
    for i in range(scale):
        print(f"random seed：{i}")
        torch.manual_seed(i)
        Model_Train(source_location,target_location)
        mae,mse,re=Model_Eval(Test_path,save_path)
        mae_record.append(mae)
        mse_record.append(mse)
        re_record.append(re)
        if mse <= min_mse_loss:
            min_mae_loss = mae
            min_mse_loss = mse
            min_re=re
            best_state = i
    print(mae_record)
    print(mse_record)
    print(re_record)
    print("MAE",np.mean(mae_record))
    print("MSE",np.mean(mse_record))
    print("MAPE",np.mean(re_record))
    print(f"Minimum MAE={min_mae_loss},Minimum MSE={min_mse_loss}，Minimum MAPE={min_re}，Best Seed={best_state}")
def main(seed,txt):

    torch.manual_seed(seed)
    Model_Train(source_location,target_location)
    Mae,Mse,Re=Model_Eval(Test_path,save_path)
    "Sava to txt"
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
    # Model training times
    max_epoch = 30
    # Get the current absolute path
    currentPath = os.getcwd().replace('\\','/')
    # source domain file path
    source_location=currentPath+'/Source_data/por/L1-L2-L3.xlsx'
    # Target domain core file path
    target_location=currentPath+'/target_data/por/A2-POR.csv'
    # The prediction result is saved in the predict_result file
    save_path="A2-POR-result.csv"
    # Target domain test file -- core data or full log
    Test_path=target_location
    opt="Adam"
    # main function
    main(seed=16,txt='A2-POR'+opt)

