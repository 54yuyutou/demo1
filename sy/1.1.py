import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))  #figsize（x,y）： 设置图形的大小，x为图形的宽， y为图形的高，单位为英寸
plt.show()

# 添加全为1的一列，准备进行矩阵运算
data.insert(0, 'Ones', 1) #添加一列名为“Ones” 值全为1的属性
print(data)

# 初始化X和y
cols = data.shape[1] #data的列数
X = data.iloc[:, 0:cols - 1]    #筛选所有行和除最后一列外的所有列
y = data.iloc[:, cols - 1:cols] #筛选所有行和最后一列



# 将X和y转化为矩阵形式，并初始化theta
X = np.matrix(X.values)  #将数组转化为矩阵
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0])) #设置初始矩阵[0 0]


# 计算代价函数
def computeCost(X, y, theta):
    m = len(y)
    J = 0
    h = X * theta.T #预测值h
    J = (1 / (2 * m)) * np.sum(np.square(h - y)) #计算实际值y和预测值h之间的平方误差和 将其乘以1 /（2 * m）作为损失函数J的值
    return J


# 输出初始的代价函数值
print("初始损失值为：", computeCost(X, y, theta))


# 梯度下降算法
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))     #创建一个形状和theta一样的全0矩阵temp，用于存储更新后的参数值。
    parameters = int(theta.ravel().shape[1])    #将theta转化为一维数组，再通过shape[1]获取其中的参数个数，赋值给parameters。
    cost = np.zeros(iters)      #创建一个长度为iters的全0数组cost，用于存储每次迭代后的代价函数值。

    m = X.shape[0]
    for i in range(iters):
        error = (X * theta.T) - y   #计算预测值与真实值之间的误差，即代价函数中的第一项。
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(term))     #对于每个参数，计算梯度，并更新对应的temp值。
        theta = temp    #将更新后的参数赋值给theta。
        cost[i] = computeCost(X, y, theta)  #计算当前迭代下的代价函数值，并存储在cost数组中。
    return theta, cost

# 初始化学习率和迭代次数，并执行梯度下降算法
alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)

computeCost(X, y, g)
# 输出最终的参数值和损失值
print("最优参数为：", g)
print("最终损失值为：", computeCost(X, y, g))

# 绘制拟合直线
x = np.linspace(data.Population.min(), data.Population.max(), 100)  #在data.Population中均匀地取100个点，存储在数组x中，用于画出拟合直线。
f = g[0, 0] + (g[0, 1] * x)     #使用训练出来的参数g，计算在x点处的拟合值，即预测人口数量对应的利润。 其中g[0,0]为截距，g[0,1]为斜率。

fig, ax = plt.subplots(figsize=(12, 8))     #创建一个12*8的图，用于画出拟合直线和训练数据的散点图。
ax.plot(x, f, 'r', label='Prediction')      #作拟合直线图，其中x为横坐标，f为纵坐标，'r'表示使用红色线条进行绘制，label='Prediction'表示将这条线的名称设置为Prediction，方便后续添加图例。
ax.scatter(data.Population, data.Profit, label='Training Data') #在画布上画出训练数据的散点图，其中data. Population为横坐标，data. Profit为纵坐标，label='Training Data'表示将这个点的名称设置为Training Data，方便后续添加图例
ax.legend(loc=2)    #将图例添加到画布上，其中loc=2表示将图例放置在左上角。
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()


from sklearn import linear_model
model = linear_model.LinearRegression() #创建一个线性回归模型，用于训练数据并预测新的数据。
X=X.tolist()
y=y.tolist()    #将X和y从数组转为列表，因为sklearn中的LinearRegression模型只接受Python列表作为输入。
model.fit(X, y) #使用模型对训练数据进行训练，即拟合出最优的模型参数。 其中X为特征矩阵，y为标签向量。 在训练过程中，模型会根据数据不断调整参数，使得预测结果与真实结果的误差最小化。

#x = np.array(X[:, 1].A1)
print(X)
x = np.array([i[1] for i in X]) #从X中提取所有行的第二个元素并保存在变量x中。将x转为数组。
f = model.predict(X).flatten()  #我们用模型对输入数据X进行预测，并将结果展平成一个一维数组。f变量存储了通过模型对X进行预测得到的结果，这也是一个一维数组。
#predict是训练后返回预测结果，是标签值     flatten()是对多维数据的降维函数。


fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()