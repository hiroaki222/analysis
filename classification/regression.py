from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

#pd.set_option('display.max_rows', None)
train = pd.read_csv('train.csv')

plt.scatter(train['PassengerId'], train['Age'])
plt.savefig(f"figure.png")
plt.clf()

print(train)
# 要らんカラムを落とす
train = train.drop('Cabin', axis=1)
train = train.drop('Name', axis=1)
train = train.drop('Ticket', axis=1)
# stringをintに
train.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
train.replace({'Embarked': {'S': 0, 'C': 1, 'Q':2}}, inplace=True)
# これは適当に中央値で埋める
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
print(train)
print(train.isnull().sum())

plt.scatter(train['PassengerId'], train['Age'])
plt.savefig(f"figure1.png")
plt.clf()