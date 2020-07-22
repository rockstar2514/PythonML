import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def LinearRegression(X,y,theta,itera,alpha):
    r,c=X.shape
    x=[]
    z=[]
    for i in range (itera):
        x.append(i+1)
        predicted=1/(1+np.exp(X@theta))
        grad=(1/r)*(np.transpose(X)@(predicted-y))
        theta=theta-alpha*grad
        error=(-1/r)*(np.transpose(y)@np.log(predicted)+np.transpose(1-y)@(np.log(1-predicted)))
        z.append(error[0][0])     
    predicted=1/(1+np.exp(theta))
    plt.plot(x,z)
    plt.show()
    print(theta)

