##### 1.感知机学习算法的原始形式
### 1.1 算法实现
Perceptron = function(X, y, eta, maxit = 5000){
  #输入：特征向量X，标签向量y，学习率eta，最大迭代次数（默认为5000）
  X = as.matrix(X)   #数据处理：将特征向量转化成矩阵形式
  
  N = dim(X)[1]  #训练样本点的个数
  n= dim(X)[2]   #特征向量的维度
  
  w = rep(0,n)   #设置初值w_0, b_0
  b = 0
  
  ID = sample(N)   #打乱样本，尽量使得每个样本点在迭代时都能被取到
  
  j = 1   #记录迭代次数
  while (1) {   
    IDx = ID[(j-1)%%N+1]   #余数运算
    #判断是否正确分类
    if(y[IDx]*(X[IDx,]%*%w +b) <= 0){   
      w = w + eta*y[IDx]*X[IDx,]   #未正确分类，进行迭代训练，更新 w 和 b
      b = b + eta*y[IDx] 
    }
    
    #设置迭代停止的条件
    if(sum(y*(X %*% w + b)<=0) == 0)   
      break    #样本点全部被正确分类时停止
    
    j = j+1
    if(j >= maxit){
      warning("Iteration does not converage.")
      break()    #迭代次数达到设置的最大值时停止
    }
  }
  
  out = list(w = w, b = b)  #输出训练的结果
  out
}

### 1.2 利用训练的模型进行预测
Perceptron.predict = function(model, X){  
  #预测需要输入的参数：训练好的感知机模型，需要预测数据的特征向量
  y = sign(X%*%model$w + model$b)    #预测
  out = y   #输出
}


### 1.3 算法检验(数据来自于教材例2.1)
X = matrix(NA,3,2)
X[1,] = c(3,3)
X[2,] = c(4,3)
X[3,] = c(1,1)  #特征向量
y = c(1,1,-1)  #标签向量

set.seed(2020211911)   #设置随机种子，保证结果的可复现性
model1 = Perceptron(X, y, eta = 1)  #训练模型
model1  #输出参数

#绘图
plot(X[,1],X[,2],pch=(y+2),col= (y+2),
    title(main = "Perceptron Model Test", xlab = "",ylab = ""))   #绘图参数
abline(a = -model1$b/(model1$w[2]+exp(-10)), b = -model1$w[1]/(model1$w[2]+exp(-10)), col = "red" )
#添加训练得到的超平面


### 1.4 判断新样本的类别
new_X = matrix(NA,3,2)  #生成测试集
new_X[1,] = c(2, 2)
new_X[2,] = c(3, 1)
new_X[3,] = c(4, 2) #新样本点的特征向量
new_Y.predict = Perceptron.predict(model1, new_X)  #预测
new_Y.predict  #输出预测类别

#绘制测试集中的样本点
for (i in 1:3) {  
  points(new_X[i,1], new_X[i,2],pch = 15, col = "blue")
}


##### 1.5 进一步讨论
library(MASS)  #加载程辑包

## 构造多元正态分布
mu1 = c(0,0)  
mu2 = c(3,3)   #均值
Sigma1 = matrix(c(1,0.5,0.5,1),ncol = 2,nrow = 2)
Sigma2 = matrix(c(1,-0.5,-0.5,1),ncol = 2,nrow = 2)  #方差

#生成两组长度为100的多元正态随机数，分别属于“正类”和“负类”
n1 = 100
n2 = 100
X1 = mvrnorm(n1, mu1, Sigma1)
X2 = mvrnorm(n2, mu2, Sigma2)   

#par(mfrow=c(2,3))   #设置绘图窗口为2行3列
#set.seed(2023) #设置随机种子，保证结果可复现

X = rbind(X1, X2)  #特征向量
y = c(rep(-1,100), rep(1,100))  #标签向量
model2 = Perceptron(X, y, eta = 1)  #训练模型
model2  #输出模型参数
#绘图
plot(X[,1],X[,2],pch=(y+2),col= (y+2),
     title(main = "Data Simulation for Perceptron", xlab = "",ylab = ""))
abline(a = -model2$b/(model2$w[2]+exp(-10)), b = -model2$w[1]/(model2$w[2]+exp(-10)), col = "red" )


#利用训练得到的模型来测试训练集
Y.hat = Perceptron.predict(model2, X)   #测试
table(y, Y.hat)    #与标志向量进行比较
sum(y == Y.hat)/length(y)   #分类正确率






##### 2.感知机学习算法的对偶形式
### 2.1 算法实现
Perceptron.Dual = function(X, y, eta, maxit = 5000){
  #输入：特征向量X，标签向量y，学习率eta，最大迭代次数（默认为5000）
  X = as.matrix(X)  #数据处理：将特征向量转化成矩阵形式
  
  N = dim(X)[1]   #训练样本点的个数
  n = dim(X)[2]   #特征向量的维度
  
  alpha = rep(0,N)  
  b = 0              #设置迭代初始值
  
  G = X %*% t(X)   #计算Gram矩阵
  
  
  ID = sample(N)    #打乱样本点，尽量使得每个样本点在迭代时都能被取到
  
  j = 1    #记录迭代次数
  while (1) {
    IDx = ID[(j-1)%%N+1]   #余数运算
    #判断是否正确分类
    if(y[IDx]*(G[IDx,]%*%(alpha*y) + b) <= 0){   #未正确分类，进行迭代训练，更新 w 和 b
      alpha[IDx] = alpha[IDx] + eta
      b = b + eta*y[IDx]
    }
    
    #设置迭代停止的条件
    if(sum(y*(G%*%(y*alpha)+b)<=0) == 0)
      break    #样本点全部被正确分类时停止
    
    j = j+1
    
    if(j >= maxit){
      warning("Iteration does not converge")
      break()    #迭代次数达到设置的最大值时停止
    }
  
  }
  w = t(X)%*%(alpha*y)    #计算 w
  out = list(w = w, b = b, alpha = alpha)   #输出训练得到的结果
  out
}

### 2.2 利用训练的模型进行预测
Perceptron.Dual.predict = function(model, X){  
  #预测需要输入的参数：训练好的感知机模型，需要预测数据的特征向量
  y = sign(X%*%model$w + model$b)    #预测
  out = y   #输出
}



### 2.3 算法检验(数据来自于教材例2.1)
X = matrix(NA,3,2)
X[1,] = c(3,3)
X[2,] = c(4,3)
X[3,] = c(1,1)  #特征向量
y = c(1,1,-1)  #标签向量

set.seed(2020211911)   #设置随机种子，保证结果的可复现性
model3 = Perceptron.Dual(X, y, eta = 1)  #训练模型
model3  #输出参数

#绘图
plot(X[,1],X[,2],pch=(y+2),col= (y+2),
     title(main = "Perceptron Model Test", xlab = "",ylab = ""))   #绘图参数
abline(a = -model3$b/(model3$w[2]+exp(-10)), b = -model3$w[1]/(model3$w[2]+exp(-10)), col = "red" )
#添加训练得到的超平面


### 2.4 判断新样本的类别
new_X = matrix(NA,3,2)  #生成测试集
new_X[1,] = c(2, 2)
new_X[2,] = c(3, 1)
new_X[3,] = c(4, 2) #新样本点的特征向量
new_Y.predict.Dual = Perceptron.Dual.predict(model3, new_X)  #预测
new_Y.predict.Dual  #输出预测类别

### 2.5 进一步讨论

library(MASS)  #加载程辑包

## 构造多元正态分布
mu1 = c(0,0)  
mu2 = c(3,3)   #均值
Sigma1 = matrix(c(1,0.5,0.5,1),ncol = 2,nrow = 2)
Sigma2 = matrix(c(1,-0.5,-0.5,1),ncol = 2,nrow = 2)  #方差

#生成两组长度为100的多元正态随机数，分别属于“正类”和“负类”
n1 = 100
n2 = 100
X1 = mvrnorm(n1, mu1, Sigma1)
X2 = mvrnorm(n2, mu2, Sigma2)   

X = rbind(X1, X2)  #特征向量
y = c(rep(-1,100), rep(1,100))  #标签向量
model4 = Perceptron.Dual(X, y, eta = 1)  #训练模型
model4  #输出模型参数

#绘图
plot(X[,1],X[,2],pch=(y+2),col= (y+2),
     title(main = "Data Simulation for Perceptron", xlab = "",ylab = ""))
abline(a = -model4$b/(model4$w[2]+exp(-10)), b = -model4$w[1]/(model4$w[2]+exp(-10)), col = "red" )

#利用训练得到的模型来测试训练集
Y.hat = Perceptron.Dual.predict(model4, X)   #测试
table(y, Y.hat)    #与标志向量进行比较
sum(y == Y.hat)/length(y)   #分类正确率

##### 3.感知机两种形式的算法比较
system.time({model5 = Perceptron(X, y, eta = 1, maxit = 10000)})
system.time({model6 = Perceptron.Dual(X, y, eta = 1, maxit = 10000)})


