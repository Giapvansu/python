import numpy as np
#chieu cao (mang 1 chieu,
h=np.array([147,150,153,155,158,160,163,165,168,170,173,175,178,180,183])
print(h)
#can nang
m=np.array([[49,50,51,52,54,56,58,59,60,72,63,64,66,67,68]]).T
print(m)
one=np.ones(np.shape(m))
print(one)
m_bar=np.concatenate((m,one),axis=1)
print(m_bar)
#cong thuc hoi quy
A=np.dot(m_bar.T,m_bar)
b=np.dot(m_bar.T,h)
print(A)
print(b)
w=np.dot(np.linalg.pinv(A),b)
print(w)
#ham so tim chieu cao dua vao can nang
def timchieucao(a):
    return w[0]*a+w[1]
print(timchieucao(80))
print("nguoi co can nang",80,"chieu cao la",timchieucao(80))
h1=np.array([75,80,100])
print("co chieu cao la:", timchieucao(h1))
###

##dung thu vien scikit-learn
import sklearn
from sklearn import datasets, linear_model
#fit model
regr=linear_model.LinearRegression()
regr.fit(m,h)
print(regr.coef_[0])