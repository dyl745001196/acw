# -*- coding: utf-8 -*-

# 线性回归例子
import numpy as np
import matplotlib.pyplot as plt


N=5

#准备数据
x=np.array([[2.104,1.600,2.400,1.416,3.000],[1,1,1,1,1]])

y=np.array([[40.0,33.0,36.9,23.2,54.0]])


theta=np.array([[1.0],[1.0]])


a=0.03


#BSG batch gradient descent批梯度下降
for i in range(100000):

	y_pred=theta.T.dot(x)

	loss=np.square(y_pred-y).sum()#二范数平方为损失函数
	
	
	grad=x.dot((y_pred-y).T)#计算梯度
	theta=theta-a*grad/N#批，每次更新参数都要把整个数据集即x过一遍
	
print loss


plt.scatter(x[0,:],y[0,:])
plt.plot(x[0,:],y_pred[0,:],color='red')
#plt.show()

theta=np.array([[1.0],[1.0]])

for i in range(10000/N):
	for j in range(N):
		y_pred=theta.T.dot(x)
		loss=np.square(y_pred-y).sum()#二范数平方为损失函数
		#每次更新只拿出一个样本来更新，SGD，stochastic
		x_entry=np.array([x[:,j]]).T
		grad=x_entry.dot(y_pred[:,j]-y[:,j])
		grad=np.array([grad])
		#print grad.T
		theta-=a*grad.T

print loss


plt.scatter(x[0,:],y[0,:])
plt.plot(x[0,:],y_pred[0,:],color='green')
plt.show()