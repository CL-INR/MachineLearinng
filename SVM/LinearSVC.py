# skl_SVM_v1a.py
# Demo of linear SVM by scikit-learn
# v1.0a: 线性可分支持向量机模型（SciKitLearn）
# Copyright 2021 YouCans, XUPT
# Crated：2021-05-15

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=40, centers=2, random_state=27)  # 产生数据集: 40个样本, 2类
modelSVM = SVC(kernel='linear', C=100)  # SVC 建模：使用 SVC类，线性核函数
# modelSVM = LinearSVC(C=100)  # SVC 建模：使用 LinearSVC类，运行结果同上
modelSVM.fit(X, y)  # 用样本集 X,y 训练 SVM 模型

print("\nSVM model: Y = w0 + w1*x1 + w2*x2") # 分类超平面模型
print('截距: w0={}'.format(modelSVM.intercept_))  # w0: 截距, YouCans
print('系数: w1={}'.format(modelSVM.coef_))  # w1,w2: 系数, XUPT
print('分类准确度：{:.4f}'.format(modelSVM.score(X, y)))  # 对训练集的分类准确度

# 绘制分割超平面和样本集分类结果
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)  # 散点图，根据 y值设置不同颜色
ax = plt.gca()  # 移动坐标轴
xlim = ax.get_xlim()  # 获得Axes的 x坐标范围
ylim = ax.get_ylim()  # 获得Axes的 y坐标范围
xx = np.linspace(xlim[0], xlim[1], 30)  # 创建等差数列，从 start 到 stop，共 num 个
yy = np.linspace(ylim[0], ylim[1], 30)  #
YY, XX = np.meshgrid(yy, xx)  # 生成网格点坐标矩阵 XUPT
xy = np.vstack([XX.ravel(), YY.ravel()]).T  # 将网格矩阵展平后重构为数组
Z = modelSVM.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])  # 绘制决策边界和分隔
ax.scatter(modelSVM.support_vectors_[:, 0], modelSVM.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')  # 绘制 支持向量
plt.title("Classification by LinearSVM (youcans, XUPT)")
plt.show()
# = 关注 Youcans，分享原创系列 https://blog.csdn.net/youcans =
