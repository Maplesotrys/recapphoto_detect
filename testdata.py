import numpy as np
 
data=np.load('data/v1/data.npz')
 
X_train=data['X_train']
Y_train=data['Y_train']
X_valid=data['X_valid']
Y_valid=data['Y_valid']
X_test= data['X_test']
Y_test= data['Y_test']
 
print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)
print(X_test.shape,Y_test.shape)
# print(X_test)