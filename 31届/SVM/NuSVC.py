# skl_SVM_v1b.py
# Demo of nonlinear SVM by scikit-learn
# v1.0b: 线性可分支持向量机模型（SciKitLearn）
# Copyright 2021 YouCans, XUPT
# Crated：2021-05-15

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.datasets import make_moons

# 数据准备：生成训练数据集，生成等高线网格数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=27) # 生成数据集
x0s = np.linspace(-1.5, 2.5, 100)  # 创建等差数列，从 start 到 stop，共 num 个
x1s = np.linspace(-1.0, 1.5, 100)  # start, stop 根据 Moon 数据范围选择确定
x0, x1 = np.meshgrid(x0s, x1s)  # 生成网格点坐标矩阵
Xtest = np.c_[x0.ravel(), x1.ravel()]  # 返回展平的一维数组
# SVC 建模，训练和输出
modelSVM1 = SVC(kernel='poly', degree=3, coef0=0.2)  # 'poly' 多项式核函数
modelSVM1.fit(X, y)  # 用样本集 X,y 训练支持向量机 1
yPred1 = modelSVM1.predict(Xtest).reshape(x0.shape)  # 用模型 1 预测分类结果
# NuSVC 建模，训练和输出
modelSVM2 = NuSVC(kernel='rbf', gamma='scale', nu=0.1)  #'rbf' 高斯核函数
modelSVM2.fit(X, y)  # 用样本集 X,y 训练支持向量机 2
yPred2 = modelSVM2.predict(Xtest).reshape(x0.shape)  # 用模型 2 预测分类结果

fig, ax = plt.subplots(figsize=(8, 6))  
ax.contourf(x0, x1, yPred1, cmap=plt.cm.brg, alpha=0.1) # 绘制模型1 分类结果
ax.contourf(x0, x1, yPred2, cmap='PuBuGn_r', alpha=0.1) # 绘制模型2 分类结果
ax.plot(X[:,0][y==0], X[:,1][y==0], "bo")  # 按分类绘制数据样本点
ax.plot(X[:,0][y==1], X[:,1][y==1], "r^")  # XUPT
ax.grid(True, which='both')
ax.set_title("Classification of moon data by LinearSVM")
plt.show()
# = 关注 Youcans，分享原创系列 https://blog.csdn.net/youcans =
