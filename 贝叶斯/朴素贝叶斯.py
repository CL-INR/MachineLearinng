#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report#导入分类报告模板

from sklearn.naive_bayes import GaussianNB#导入先验概率的高斯朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB#导入先验概率为多项式分布的朴素贝叶斯模型
from sklearn.naive_bayes import BernoulliNB#导入先验概率为伯努利分布的朴素贝叶斯模型



#1.导入sklearn自带的数据集：威斯康星乳腺肿瘤数据集（load_breast_cancer）。
breast_cancer=load_breast_cancer()

#2.打印数据集键值（keys），查看数据集包含的信息。
print(breast_cancer.keys())

#3.打印查看数据集中标注好的肿瘤分类（target_names）、肿瘤特征名称（feature_names）。
print(breast_cancer.target_names)
print(breast_cancer.feature_names)

#4.将数据集拆分为训练集和测试集，打印查看训练集和测试集的数据形态（shape）。
data=pd.DataFrame(breast_cancer.data)
target=pd.DataFrame(breast_cancer.target)
X=np.array(data.values)
y=np.array(target.values)
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
print(train_X.shape,test_X.shape)

#5.配置高斯朴素贝叶斯模型。
model=GaussianNB()

#6.训练模型。
model.fit(train_X,train_y)

#7.评估模型，打印查看模型评分（分别打印训练集和测试集的评分）。
#cross_val_score交叉验证
#计算高斯朴素贝叶斯算法模型的准确率
sorce=cross_val_score(model,train_X,train_y,cv=10,scoring='accuracy')
print("高斯朴素贝叶斯模型的准确率:",sorce.mean())
#打印训练集和测试集的评分
print("高斯朴素贝叶斯模型训练集的评分:",model.score(train_X,train_y))
print("高斯朴素贝叶斯模型测试集的评分:",model.score(test_X,test_y))


#8.模型预测：选取某一样本进行预测。（可以进行多次不同样本的预测）
pre_y=model.predict(test_X)
# sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
print(classification_report(test_y,pre_y))

