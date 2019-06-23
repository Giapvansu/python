import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets.mldata import fetch_mldata
import matplotlib.pyplot as plt
#from display_network import *
mnist = fetch_mldata('mnist-original', data_home='/media/Vancouver/apps/mnist_dataset/')
print(mnist)
X_all=mnist.data
y_all=mnist.target
X0 = X_all[np.where(y_all == 0)[0]] # all digit 0
X1 = X_all[np.where(y_all == 1)[0]] # all digit 1
y0 = np.zeros(X0.shape[0]) # class 0 label
y1 = np.ones(X1.shape[0]) # class 1 label
X = np.concatenate((X0, X1), axis = 0) # all digits
y = np.concatenate((y0, y1)) # all labels
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000)
#################
print(X_train.shape)
model=LogisticRegression(C=1e5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(y_pred)
print("Accuracy %.2f%%" % (100*accuracy_score(y_test,y_pred.tolist())))
mis=np.where((y_pred-y_test) !=0)[0]
print(mis)
Xmis=X_test[mis,:]
print(Xmis.shape)
#in anh sai

#filename = 'mnist_mis.pdf'
#with PdfPages(filename) as pdf:
#plt.axis('off')
#A = display_network(Xmis.T, 1, Xmis.shape[0])
#f2 = plt.imshow(A, interpolation='nearest' )
#plt.gray()
#pdf.savefig(bbox_inches='tight')
#plt.show()