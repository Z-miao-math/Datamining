# 直接利用循环
myknn1 = function(X.train, y.train, X.test, k=3, p=2){
  # 输入: 训练数据的特征及标签，测试数据集的特征，近邻数（默认为3），距离参数（默认p=2为欧式距离）
  n = dim(X.test)[1]
  pred = rep(NA,n)  #设置最终结果的储存位置
  for (i in 1:n) {
    distances = rep(0, length(nrow(X.train)))   #设置距离的储存位置
    for (j in 1:nrow(X.train)) {
      dist = 0
      for (d in 1:ncol(X.train)) {
        dist = dist + abs(X.train[j,d] - X.test[i,d])^p   
      }
      distances[j] = dist^(1/p)    #计算距离
    }
    nn = order(distances)[1:k]   # 找近邻
    class.frequency = table(y.train[nn])   #统计频数
    most.frequency.classes = names(class.frequency)[class.frequency == max(class.frequency)]   
    # 提取最大频数,将其对应的类别作为预测的分类结果
    
    pred[i] = sample(most.frequency.classes, 1)   #若有多个最大频数，随机选取对应的一类作为预测结果
  }
  out = factor(pred, levels = levels(y.train))   # 以因子形式输出结果
  out
}



# 直接利用循环（2）
myknn3 = function(X.train, y.train, X.test, k=3, p=2){
  # 输入: 训练数据的特征及标签，测试数据集的特征，近邻数（默认为3），距离参数（默认p=2为欧式距离）
  n = dim(X.test)[1]
  m = dim(X.train)[1]
  pred = rep(NA, n)   #预测结果的储存位置
  
  for(i in 1:n){
    distances = numeric(m)   # 设置距离的储存位置
    for (j in 1:m) {   
      distances[j] = sum(abs(X.train[j,]-X.test[i,])^p)^(1/p)
      # 计算第i个测试样本点到第j个训练样本点的距离
    }
    nn = order(distances)[1:k]   #第i个测试样本点的k个近邻
    
    class.frequency = table(y.train[nn])   # 每一类的频率
    most.frequency.classes = names(class.frequency)[class.frequency == max(class.frequency)]
    # 确定最大频率所对应的类别
    pred[i] = sample(most.frequency.classes, 1)
    # 多数表决：最大频数对应的类别即为第i个样本点的预测类别。
    # 若有多个最大频数，随机选取对应的一类作为预测结果
  }
  out = factor(pred, levels = levels(y.train))
  out  #以因子形式输出预测结果
}

# 利用 apply 函数
myknn2 = function(X.train, y.train, X.test, k=3, p=2){
  # 输入: 训练数据的特征及标签，测试数据集的特征，近邻数（默认为3），距离参数（默认p=2为欧式距离）
  n = dim(X.test)[1]
  pred = rep(NA, n)  #预测结果的储存位置
  
  for(i in 1:n){
    nn = order(apply(X.train, 1, function(x) (sum((abs(x - X.test[i,]))^p))^(1/p)))[1:k]
    # 计算第 i个测试样本点到每个训练样本点的距离并求出k个近邻
    class.frequency = table(y.train[nn])  #统计每个类别的频数
    most.frequency.classes = names(class.frequency)[class.frequency == max(class.frequency)]
    # 将最大频数对应的类作为预测结果
    pred[i] = sample(most.frequency.classes, 1)
    # 若有多个最大频数，随机选取对应的一类作为预测结果
  }
  out = factor(pred, levels = levels(y.train))
  out   #输出预测结果
}


# 仅预测一个样本点的函数
myknn.one = function(X.test, X.train, y.train, k=3, p=2){
  dist.one = apply(X.train, 1, function(x)(sum((abs(x - X.test))^p))^(1/p))
  nn = order(dist.one)[1:k]
  class.frequency = table(y.train[nn])
  most.frequency.classes = names(class.frequency)[class.frequency == max(class.frequency)]
  pred = sample(most.frequency.classes,1)
}


# 预测多个样本点
myknn = function(X.train, y.train, X.test, k=3, p=2){
  pred = apply(X.test, 1, myknn.one, X.train, y.train, k, p)
  out = factor(pred, levels = levels(y.train))
  out 
}



# 5-折交叉验证
knn.CV = function(X, y, fold=5, k=3, p=2){
  
  n = dim(X)[1]
  CV.ID = sample(rep(1:fold,length.out=n))
  CV.err = rep(NA,fold)
  for (j in 1:fold) {
    X.train = X[CV.ID!=j,]
    X.test = X[CV.ID==j,]
    y.train = y[CV.ID!=j]
    y.test = y[CV.ID==j]
    pred = myknn(X.train, y.train, X.test, k=k, p=p)
    CV.err[j] = sum(pred!=y.test)/length(y.test)
  }
  CV.err.mean = mean(CV.err)
  CV.err.sd = sd(CV.err)
  
  out =list(CV.err=CV.err, CV.err.mean=CV.err.mean, CV.err.sd=CV.err.sd)
  out
}


# 交叉验证确定k的取值
knn.selection = function(X, y, fold=5, Klist=seq(1,9,2), p=2){
  
  KK=length(Klist)
  err = rep(NA,KK)
  for (j in 1:KK) {
    err[j] = knn.CV(X, y, fold = 5, k = Klist[j], p = p)$CV.err.mean
  }
  out = list(Klist = Klist, CV.err = err)
  out
}


# 利用数据集验证
library("class")
iris = iris
ID = sample(dim(iris)[1])
train.number = 100
X.train = iris[ID[1:train.number],1:4]
X.test = iris[ID[(train.number+1):150],1:4]
y.train = iris[ID[1:train.number],5]
y.test = iris[ID[(train.number+1):150],5]

# class程辑包包括的函数
knn0 = knn(X.train, X.test, y.train, k = 3, prob = TRUE)
knn0

knn1 = myknn1(X.train, y.train, X.test, k = 3, p = 2)
knn1

knn2 = myknn2(X.train, y.train, X.test, k = 3, p = 2)
knn2

knn3 = myknn3(X.train, y.train, X.test, k = 3, p = 2)
knn3

cbind(knn0, knn2)

sum(knn0==knn1)
sum(knn0==knn2)
sum(knn0==knn3)
table(knn2, y.test)
table(knn0, y.test)
table(knn1, y.test)
table(knn3, y.test)

plot(X.train[,1],X.train[,2],col = y.train)
points(X.test[,1],X.test[,2],col = y.train, pch=0)


# 从网页读取数据
iris1 <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",", header=FALSE)

iris1 = iris1

ID = sample(dim(iris1)[1])

train.number = 100
X.train = iris1[ID[1:train.number],1:4]
X.test = iris1[ID[(train.number+1):150],1:4]

y.train = as.factor(iris1[ID[1:train.number],5])
y.test = as.factor(iris1[ID[(train.number+1):150],5])
  
# class程辑包包括的函数
knn0 = knn(X.train, X.test, y.train, k = 3, prob = TRUE)
knn0


knn1 = myknn1(X.train, y.train, X.test, k = 3, p = 2)
knn1

knn2 = myknn2(X.train, y.train, X.test, k = 3, p = 2)
knn2

knn3 = myknn3(X.train, y.train, X.test, k = 3, p = 2)
knn3

cbind(knn0, knn2)

sum(knn0==knn1)
sum(knn0==knn2)
sum(knn0==knn3)
table(knn2, y.test)
table(knn0, y.test)
table(knn1, y.test)
table(knn3, y.test)

# 比较运行时间
time1 = system.time({knn1 = myknn(X.train, y.train, X.test, k = 3, p = 2)})
time2 = system.time({knn2 = myknn1(X.train, y.train, X.test, k = 3, p = 2)})
time3 = system.time({knn3 = myknn2(X.train, y.train, X.test, k = 3, p = 2)})
time4 = system.time({knn4 = myknn3(X.train, y.train, X.test, k = 3, p = 2)})
time1
time2
time3
time4
table(knn1,knn2)
sum(knn1==knn2)

# 交叉验证选择k值
CV.err = knn.CV(iris[,1:4], iris[,5], k=5)

CV = knn.selection(iris[,1:4], iris[,5], fold = 5, Klist = seq(1,20,1),p = 2)
plot(CV$Klist, CV$CV.err, type = 'o')



# 可视化
# install.packages("umap")
library(umap)
iris.data = iris[,grep("Sepal|Petal", colnames(iris))]
iris.labels = iris[,"Species"]

iris.umap = umap(iris.data)
plot(iris.umap$layout[,1], iris.umap$layout[,2], col=iris.labels)


X.train0 = iris.umap$layout[ID[1:train.number],]
X.test0 = iris.umap$layout[ID[(train.number+1):150],]

y.train0 = iris.labels[ID[1:train.number]]
X.test0 = iris.labels[ID[(train.number+1):150]]

