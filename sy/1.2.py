import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''from 1.1 import gradientDescent
from 1.1 import computeCost'''

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head() #head()函数:打印数据表但由于默认设置只能打印前5行。


data2 = (data2 - data2.mean()) / data2.std()
data2.head()
print(data2)

data2.insert(0, 'Ones', 1)

cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]



X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
alpha = 0.01
epoch = 1000
theta2 = np.matrix(np.array([0, 0, 0]))

def computeCost(X2, y2, theta2):
    m = len(y2)
    J = 0
    h = X2 * theta2.T #预测值h
    J = (1 / (2 * m)) * np.sum(np.square(h - y2)) #计算实际值y和预测值h之间的平方误差和 将其乘以1 /（2 * m）作为损失函数J的值
    return J

def gradientDescent(X2, y2, theta2, alpha,epoch ):
    temp = np.matrix(np.zeros(theta2.shape))     #创建一个形状和theta一样的全0矩阵temp，用于存储更新后的参数值。
    parameters = int(theta2.ravel().shape[1])    #将theta转化为一维数组，再通过shape[1]获取其中的参数个数，赋值给parameters。
    cost = np.zeros(epoch)      #创建一个长度为iters的全0数组cost，用于存储每次迭代后的代价函数值。

    m = X2.shape[0]
    for i in range(epoch):
        error = (X2 * theta2.T) - y2   #计算预测值与真实值之间的误差，即代价函数中的第一项。
        for j in range(parameters):
            term = np.multiply(error, X2[:, j])
            temp[0, j] = theta2[0, j] - ((alpha / m) * np.sum(term))     #对于每个参数，计算梯度，并更新对应的temp值。
        theta2 = temp    #将更新后的参数赋值给theta。
        cost[i] = computeCost(X2, y2, theta2)  #计算当前迭代下的代价函数值，并存储在cost数组中。
    return theta2, cost



g2, cost2 = gradientDescent(X2, y2, theta2, alpha, epoch)
print(computeCost(X2, y2, g2)), g2

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost2, 'r')  # np.arange()自动返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

def NormalEquations(X,y):
    theta = np.linalg.inv(X.T*X)*X.T*y
    return theta

final_theta2=NormalEquations(X2, y2)