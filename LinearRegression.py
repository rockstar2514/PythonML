import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def LinearRegression(X,y,theta,itera,alpha):
    r,c=X.shape
    x=[]
    z=[]
    for i in range (itera):
        x.append(i+1)
        predicted=X@theta
        grad=(1/r)*(np.transpose(X)@(predicted-y))
        theta=theta-alpha*grad
        error=(1/(2*r))*np.transpose(predicted-y)@(predicted-y)
        z.append(error[0][0])#error is a matrix     
    predicted=X@theta
    plt.plot(x,z)
    plt.show()
    print(theta)    
xdata=np.arange(0,4,0.01)
ydata=3.86718*xdata*xdata+1.5+0.2*np.random.randn(len(xdata))
plt.scatter(xdata,ydata)
plt.show()
X=np.ones((400,3))
for i in range(400):
    X[i][1]=xdata[i]
for i in range(400):
    X[i][2]=xdata[i]*xdata[i]; 
theta=np.ones((3,1))
theta[0]=1
theta[1]=1
theta[2]=1
y=np.ones((len(ydata),1))
for i in range(len(ydata)):
    y[i][0]=ydata[i]
LinearRegression(X,y,theta,100,0.01)
#very important to convert lists to numpy arrays
#Our function LinearRegression() takes X->the design matrix, y->the target values, theta->initial values of theta, itera-> number of iterations, alpha-> learning rate.(all are numpy arrays)
