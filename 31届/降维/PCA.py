import numpy as np

class CPCA():
    def __init__(self, X, dim):
        self.x = X
        self.k = dim
        self.mx = self.Mean()
        self.Covx = self.Cov()

    def Mean(self):
        l, w = self.x.shape
        x = self.x
        print(x.shape, l, w, x[9][2])
        sum1 = sum2 = sum3 = 0
        for i in range(l):
            sum1 = sum1 + x[i][0]
            sum2 = sum2 + x[i][1]
            sum3 = sum3 + x[i][2]
        mx1 = sum1 / l
        mx2 = sum2 / l
        mx3 = sum3 / l
        a = [mx1, mx2, mx3]
        return a

    
    def Cov(self):
        x = self.x
        l, w = x.shape
        for i in range(l):
            x[i] = x[i] - self.mx
        Covx = np.dot(x.T, x) / l
        return Covx

    def Eig(self):
        x = self.Covx
        a, b = np.linalg.eig(x)  # 特征值赋值给a，对应特征向量赋值给b。
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.k)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.k, U)
        return U

    def forward(self):
        u = self.Eig()
        pcad = np.dot(self.x, u)
        return pcad

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    print(X.shape)
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    print("降维后：\n", CPCA(X, 2).forward())
