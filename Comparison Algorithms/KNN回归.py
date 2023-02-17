import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def mean_relative_error(y_true, y_pred,):
    """MAPE"""
    relative_error = np.average(np.abs((y_true - y_pred) / y_true)*100, axis=0)
    return relative_error
currentPath = os.getcwd().replace('\\', '/')
all_data=pd.read_excel(currentPath+'/Source_data/por/L1-L2-L3.xlsx')
test_data=pd.read_csv(currentPath+'/target_data/por/A2-POR.csv')
source_test_size=61/711
all_data.head()
all_data=all_data.dropna(axis=1)
test_data=test_data.dropna(axis=1)

print(f"Number of source domains:{all_data.shape}")
all_data=all_data.iloc[:,1:].values
test_data=test_data.iloc[:,1:].values
x=all_data[:,:-1]
y=all_data[:,-1]
test_x=test_data[:,:-1]
test_y=test_data[:,-1]
print(f"Number of target domains:{test_x.shape},{test_y.shape}")

"""Standardization"""
scaler = StandardScaler()
x=scaler.fit_transform(x)
test_x=scaler.fit_transform(test_x)
"""Split data"""
x,x_D,y,y_D = train_test_split(x,y,test_size=source_test_size,random_state=55)
"""Build model"""
k = 5
knn = KNeighborsRegressor()
knn.fit(x, y)
result = knn.predict(test_x)
"""Result display"""
print(f"Mean square error of target domain：{mean_squared_error(result,test_y)}")
print(f"Average absolute error of target domain：{mean_absolute_error(result,test_y)}")
print(f"Mean absolute percentage error of target domain：{mean_relative_error(test_y,result)}%")

# result_source = knn.predict(x_D)
# print(f"Mean square error of source domain：{mean_squared_error(result_source,y_D)}")
# print(f"Average absolute error of source domain：{mean_absolute_error(result_source,y_D)}")
# print(f"Mean absolute percentage error of source domain：{mean_relative_error(y_D,result_source)}%")

