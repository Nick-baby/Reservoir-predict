"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
all_data=pd.read_excel(os.getcwd().replace('\\','/')+'/Source_data/por/N1_oneclass_svm.xlsx').values[200:300,-2:]
all_data=StandardScaler().fit_transform(all_data)
xx, yy = np.meshgrid(np.linspace(-7.5, 5, 500), np.linspace(-7.5, 7.5, 500))

X_train =all_data[:,-2:]
# fit the model
clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)

n_error_train = y_pred_train[y_pred_train == -1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="blueviolet", s=s, edgecolors="k")
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((-7.5, 5))
plt.ylim((-7.5, 7.5))
plt.xticks(())
plt.yticks(())
plt.xlabel(
    "DEN\n "
    "error number: %d"
    % (n_error_train)
)
plt.ylabel(
    "U error train: %d/%d "
)
plt.show()
