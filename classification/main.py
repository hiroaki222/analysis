from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

df = pd.read_csv('train.csv')
ans = pd.read_csv('test.csv')

# 要らんカラムを落とす
df = df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]]

# stringをintに
df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'Embarked': {'S': 0, 'C': 1, 'Q':2}}, inplace=True)
# これは適当に中央値で埋める
df['Embarked'].fillna(df['Embarked'].median(), inplace=True)
# PclassごとのAgeの平均値で埋める
df['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.mean()))['Age']
