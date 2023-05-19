from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import random

train = pd.read_csv('train.csv')
train = train.dropna(subset=['Cabin'], axis=0)

train.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
train.replace({'Embarked': {'S': 0, 'C': 1, 'Q':2}}, inplace=True)

train['Embarked'].fillna(train['Embarked'].median(), inplace=True)


#cols = [i for i in train.columns if train[i].isnull().any()]
# 欠損値がないデーター
trainData = train.loc[train['Age'].notnull()]
# 欠損値のあるデータ
testData = train.loc[train['Age'].isnull()]
# 説明変数と目的変数を分離
XTrain = trainData.drop('Age', axis=1)
yTrain = trainData['Age']
XTest = testData.drop('Age', axis=1)
# 回帰
model = LinearRegression()
model.fit(XTrain, yTrain)

# 欠損値予測
y_pred = model.predict(XTest)
# 保管
train.loc[train['Age'].isnull(), 'Age'] = y_pred
print(train.isnull().sum())
""" X = train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = train['Age'].values """
