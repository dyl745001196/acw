
# -*- coding: utf-8 -*-
import 	numpy as np 
import matplotlib.pyplot as plt 





# N is the batch size ,D_in is the input dimension
# H is the hidden dimension ,D_out is the ouput dimension

N,D_in,H,D_out=64,1000,100,100

# create random input and output data

x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)


# randomly initialize weights 

w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)



learning_rate=1e-6

for t in range(200):

	#forward pass前向传播，计算模型预测的y_pred
	h=x.dot(w1)
	h_relu=np.maximum(h,0)#修成线性单元
	y_pred=h_relu.dot(w2)


	#计算并且输出损失
	loss=np.square(y_pred-y).sum()


	print(t,loss)


	grad_y_pred=2.0*(y_pred-y)

	grad_w2=h_relu.T.dot(grad_y_pred)

	grad_h_relu = grad_y_pred.dot(w2.T)

	grad_h=grad_h_relu.copy()
	grad_h[h<0]=0

	grad_w1=x.T.dot(grad_h)


	#更新w1,w2
	w1-=learning_rate*grad_w1
	w2-=learning_rate*grad_w2



