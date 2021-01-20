#!/usr/bin/env python
# coding: utf-8


from collections import Counter
import pandas as pd  # 数据处理
import numpy as np  # 数学运算
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.metrics import accuracy_score  # 准确率函数

RANDOM_SEED = 2020  # 固定随机种子

csv_data = './data/high_diamond_ranked_10min.csv'  # 数据路径
data_df = pd.read_csv(csv_data, sep=',')  # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId')  # 舍去对局标号列

print(data_df.iloc[0])  # 输出第一行数据
data_df.describe()  # 每列特征的简单统计信息

drop_features = ['blueGoldDiff', 'redGoldDiff',
                 'blueExperienceDiff', 'redExperienceDiff',
                 'blueCSPerMin', 'redCSPerMin',
                 'blueGoldPerMin', 'redGoldPerMin']  # 需要舍去的特征列
df = data_df.drop(columns=drop_features)  # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')]  # 取出要作差值的特征名字（除去red前缀）
for info in info_names:  # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info]  # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood'])  # 原有的FirstBlood可删除

discrete_df = df.copy()  # 先复制一份数据
for c in df.columns[1:]:  # 遍历每一列特征，跳过标签列
    '''
    请离散化每一列特征，即discrete_df[c] = ...

    提示：
    对于有些特征本身取值就很少，可以跳过即 if ... : continue
    对于其他特征，可以使用等区间离散化、等密度离散化或一些其他离散化方法
    可参考使用pandas.cut或qcut
    '''

all_y = discrete_df['blueWins'].values  # 所有标签数据
feature_names = discrete_df.columns[1:]  # 所有特征的名称
all_x = discrete_df[feature_names].values  # 所有原始特征值，pandas的DataFrame.values取出为numpy的array矩阵

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape  # 输出数据行列信息


# ###  决策树模型的实现
# ***本小节要求实现决策树模型，请补全算法代码***

# In[ ]:


# 定义决策树类
class DecisionTree(object):
    def __init__(self, classes, features,
                 max_depth=10, min_samples_split=10,
                 impurity_t='entropy'):
        '''
        传入一些可能用到的模型参数，也可能不会用到
        classes表示模型分类总共有几类
        features是每个特征的名字，也方便查询总的共特征数
        max_depth表示构建决策树时的最大深度
        min_samples_split表示构建决策树分裂节点时，如果到达该节点的样本数小于该值则不再分裂
        impurity_t表示计算混杂度（不纯度）的计算方式，例如entropy或gini
        '''
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None  # 定义根节点，未训练时为空

    '''
    请实现决策树算法，使得fit函数和predict函数可以正常调用，跑通之后的测试代码，
    要求之后测试代码输出的准确率大于0.6。

    提示：
    可以定义额外一些函数，例如
    impurity()用来计算混杂度
    gain()调用impurity用来计算信息增益
    expand_node()训练时递归函数分裂节点，考虑不同情况
        1. 无需分裂 或 达到分裂阈值
        2. 调用gain()找到最佳分裂特征，递归调用expand_node
        3. 找不到有用的分裂特征
        fit函数调用该函数返回根节点
    traverse_node()预测时遍历节点，考虑不同情况
        1. 已经到达叶节点，则返回分类结果
        2. 该特征取值在训练集中未出现过
        3. 依据特征取值进入相应子节点，递归调用traverse_node
    当然也可以有其他实现方式。

    '''

    def fit(self, feature, label):
        assert len(self.features) == len(feature[0])  # 输入数据的特征数目应该和模型定义时的特征数目相同
        '''
        训练模型
        feature为二维numpy（n*m）数组，每行表示一个样本，有m个特征
        label为一维numpy（n）数组，表示每个样本的分类标签

        提示：一种可能的实现方式为
        self.root = self.expand_node(feature, label, depth=1) # 从根节点开始分裂，模型记录根节点
        '''

    def predict(self, feature):
        assert len(feature.shape) == 1 or len(feature.shape) == 2  # 只能是1维或2维
        '''
        预测
        输入feature可以是一个一维numpy数组也可以是一个二维numpy数组
        如果是一维numpy（m）数组则是一个样本，包含m个特征，返回一个类别值
        如果是二维numpy（n*m）数组则表示n个样本，每个样本包含m个特征，返回一个numpy一维数组

        提示：一种可能的实现方式为
        if len(feature.shape) == 1: # 如果是一个样本
            return self.traverse_node(self.root, feature) # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature]) # 如果是很多个样本
        '''


# 定义决策树模型，传入算法参数
DT = DecisionTree(classes=[0, 1], features=feature_names, max_depth=5, min_samples_split=10, impurity_t='gini')

DT.fit(x_train, y_train)  # 在训练集上训练
p_test = DT.predict(x_test)  # 在测试集上预测，获得预测值
print(p_test)  # 输出预测值
test_acc = accuracy_score(p_test, y_test)  # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc))  # 输出准确率

# ### 模型调优
# 第一次模型测试结果可能不够好，可以先检查调试代码是否有bug，再尝试调整参数或者优化计算方法。

# ### 总结
# 一个完整的机器学习任务包括：确定任务、数据分析、特征工程、数据集划分、模型设计、模型训练和效果测试、结果分析和调优等多个阶段，本案例以英雄联盟游戏胜负预测任务为例，给出了每个阶段的一些简单例子，帮助大家入门机器学习，希望大家有所收获！
