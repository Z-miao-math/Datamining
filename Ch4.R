naive_bayes <- function(train_data, train_labels, test_data) {
  
  # 将训练数据和测试数据合并
  data = rbind(train_data, test_data)
  
  # 根据数据类型识别离散特征和连续特征
  discrete_feats = sapply(data, function(x) is.factor(x) || is.logical(x))
  continuous_feats = sapply(data, function(x) is.numeric(x) && !is.factor(x) && !is.logical(x))
  
  # 计算每个类别的先验概率
  classes = unique(train_labels)
  prior_probs = sapply(classes, function(c) sum(train_labels == c) / length(train_labels))
  
  # 计算每个特征在每个类别下的条件概率
  conditional_probs = list()
  for (i in 1:length(classes)) {
    c = classes[i]
    conditional_probs[[i]] = list()
    for (j in 1:ncol(data)) {
      if (discrete_feats[j]) {   #第j列为离散特征
        # 离散特征的条件概率
        conditional_probs[[i]][[j]] = table(train_data[,j][train_labels == c]) / sum(train_labels == c)
      } else if (continuous_feats[j]) {
        # 连续特征的条件概率
        mu = mean(train_data[,j][train_labels == c])
        sigma = sd(train_data[,j][train_labels == c])
        conditional_probs[[i]][[j]] = list(mu = mu, sigma = sigma)
      }
    }
  }
  
  # 对测试数据进行分类
  preds = rep(NA, nrow(test_data))
  for (i in 1:nrow(test_data)) {
    row = test_data[i,]
    probs = rep(0, length(classes))
    for (j in 1:length(classes)) {
      prob = prior_probs[j]
      for (k in 1:ncol(data)) {
        if (discrete_feats[k]) {
          prob = prob * conditional_probs[[j]][[k]][row[k]]
        } else if (continuous_feats[k]) {
          prob = prob * dnorm(as.numeric(row[k]), mean = conditional_probs[[j]][[k]]$mu, sd = conditional_probs[[j]][[k]]$sigma)
        }
      }
      probs[j] = prob
    }
    preds[i] = classes[which.max(probs)]
  }
  
  # 返回结果
  out = list(classes=classes, preds=preds)
  out
}






# laplace平滑
naive_bayes = function(train_data, train_labels, test_data, lambda = 0) {
  #输入：训练数据的特征、标签，测试数据集的特征
  # 将训练数据和测试数据合并
  data = rbind(train_data, test_data)
  
  # 根据数据类型识别离散特征和连续特征
  discrete_feats = sapply(data, function(x) is.factor(x) || is.logical(x))
  continuous_feats = sapply(data, function(x) is.numeric(x) && !is.factor(x) && !is.logical(x))
  
  # 计算每个类别的先验概率
  classes = unique(train_labels)   #类别
  prior_probs = sapply(classes, function(c) (sum(train_labels == c) +lambda) / (length(train_labels)) )  #统计每一类的频率
  
  # 计算每个特征在每个类别下的条件概率
  conditional_probs = list()  #设置储存结果的列表
  for (i in 1:length(classes)) {
    c = classes[i]
    conditional_probs[[i]] = list()
    for (j in 1:ncol(data)) {
      if (discrete_feats[j]) {   #第j列为离散特征
        # 离散特征的条件概率（用频率估计）
        conditional_probs[[i]][[j]] = (table(train_data[,j][train_labels == c]) + lambda) / (sum(train_labels == c) + lambda * length(unique(data[,j])))
      } else if (continuous_feats[j]) {
        # 连续特征的条件概率（求出正态分布的均值与标准差）
        mu = mean(train_data[,j][train_labels == c])   #均值
        sigma = sd(train_data[,j][train_labels == c])  #标准差
        conditional_probs[[i]][[j]] = list(mu = mu, sigma = sigma)
      }
    }
  }
  
  # 对测试数据进行分类
  preds = rep(NA, nrow(test_data))  #设置预测结果储存位置
  for (i in 1:nrow(test_data)) {
    row = test_data[i,]
    probs = rep(0, length(classes))  # 后验概率储存位置
    for (j in 1:length(classes)) {
      prob = prior_probs[j]  #先验概率作为初始值
      for (k in 1:ncol(data)) {
        if (discrete_feats[k]) {  
          #离散特征后验概率等于先验概率乘以条件概率
          prob = prob * conditional_probs[[j]][[k]][row[k]]
        } else if (continuous_feats[k]) {
          #连续特征后验概率等于先验概率乘以相应正态分布的概率密度
          prob = prob * dnorm(as.numeric(row[k]), mean = conditional_probs[[j]][[k]]$mu, sd = conditional_probs[[j]][[k]]$sigma)
        }
      }
      probs[j] = prob
    }
    preds[i] = classes[which.max(probs)]  #概率最大值对应的类别作为预测的结果
  }
  
  # 输出结果：包括类别和预测的结果
  out = list(classes=classes,preds=preds)
  out
}





iris1 <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",", header=FALSE)

iris1 = iris1

ID = sample(dim(iris1)[1])

train.number = 100
train_data = iris1[ID[1:train.number],1:4]
test_data = iris1[ID[(train.number+1):150],1:4]

train_labels = iris1[ID[1:train.number],5]
test_labels = iris1[ID[(train.number+1):150],5]

pred_labels = naive_bayes(train_data, train_labels, test_data, lambda = 0)
pred_labels


sum(pred_labels$preds==test_labels)

library(e1071)
model=naiveBayes(train_data, train_labels,0)
pred_labels0 = predict(model,test_data)
sum(pred_labels$preds == pred_labels0)




statlog = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat", sep=" ", header=FALSE)
set.seed(2020)
ID = sample(dim(statlog)[1])

train.number = 210
train_data = statlog[ID[1:train.number],1:13]
test_data = statlog[ID[(train.number+1):270],1:13]

train_labels = statlog[ID[1:train.number],14]
test_labels =statlog[ID[(train.number+1):270],14]
a = naive_bayes(train_data, train_labels, test_data)
a

sum(a$preds==test_labels)


###
tae = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data", sep=",", header=FALSE)
ID = sample(dim(tae)[1])

train.number = 100
train_data = tae[ID[1:train.number],1:5]
test_data = tae[ID[(train.number+1):151],1:5]

train_labels = tae[ID[1:train.number],6]
test_labels =tae[ID[(train.number+1):151],6]
a = naive_bayes(train_data, train_labels, test_data)
a

test_labels 


library(e1071)
model=naiveBayes(train_data, train_labels, 0)
b = predict(model,test_data)





###
iris1 <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",", header=FALSE)

iris1 = iris1

ID = sample(dim(iris1)[1])

train.number = 100
train_data = iris1[ID[1:train.number],1:4]
test_data = iris1[ID[(train.number+1):150],1:4]

train_labels = iris1[ID[1:train.number],5]
test_labels = iris1[ID[(train.number+1):150],5]
a =naive_bayes(train_data, train_labels, test_data)
a




##### 5-折交叉验证函数
naive_CV = function(data, labels, fold=5){
  n = dim(data)[1]
  set.seed(2020211911)
  CV.ID = sample(rep(1:fold,length.out=n))
  CV.err = rep(NA, fold)
  for (j in 1:fold) {
    train_data = data[CV.ID!=j,]
    test_data = data[CV.ID==j,]
    train_labels = labels[CV.ID!=j]
    test_labels = labels[CV.ID==j]
    pred = naive_bayes(train_data, train_labels, test_data)
    CV.err[j] = sum(pred$preds!=test_labels)/length(test_labels)
  }
  CV.err.mean = mean(CV.err)
  CV.err.sd = sd(CV.err)
  
  plot(1:fold, CV.err, type = 'l')
  out =list(CV.err=CV.err, CV.err.mean=CV.err.mean, CV.err.sd=CV.err.sd)
  out
}



iris1 = iris1
data = iris1[,1:4]
labels = iris1[,5]
naive_CV(data, labels)














