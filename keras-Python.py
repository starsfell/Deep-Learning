
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:25:04 2018

@author: xintong.yan
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets
import matplotlib as plt

# 对于具有2个类的单输入模型（二进制分类）：

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))    # 两分类变量,dense=1
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=1000, n_features=100, centers=2, cluster_std=1)

# data visualization
plt.pyplot.scatter(X[:,0], X[:,1], c=y)


#将标签转为 one-hot 编码
one_hot_label = keras.utils.to_categorical(y, num_classes=2)



# 训练模型，以500个样本为一个 batch 进行迭代
model.fit(X, y, epochs=10, batch_size=500)




# 对于具有10个类的单输入模型（多分类分类）：
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))   # input_dim=自变量的数量
model.add(Dense(10, activation='softmax'))    # 一共有10个类
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=1000, n_features=100, centers=10, cluster_std=1)

# 将标签转换为分类的 one-hot 编码
one_hot_labels = keras.utils.to_categorical(y, num_classes=10)

# 训练模型，以 32 个样本为一个 batch 进行迭代
model.fit(X, one_hot_labels, epochs=10, batch_size=32)



