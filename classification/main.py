from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Activation, BatchNormalization, Dense
from keras.layers.core import Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pandasDataFrameをすべて表示するための設定
pd.set_option('display.max_rows', None)

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# stringをintに
df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'Embarked': {'S': 0, 'C': 1, 'Q':2}}, inplace=True)
test.replace({'Embarked': {'S': 0, 'C': 1, 'Q':2}}, inplace=True)
# これは適当に中央値で埋める
df['Embarked'].fillna(df['Embarked'].median(), inplace=True)
test['Embarked'].fillna(test['Embarked'].median(), inplace=True)

# PclassごとのAgeの平均値で埋める
df['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.mean()))['Age']
test['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.mean()))['Age']

# DataFrameを結合
all = pd.concat([df, test], sort = False)

# testのPassengerIdを格納
index = test[['PassengerId']]

# 要らんカラムを落とす
df = df[['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare']]
test = test[['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare']]


# カテゴリ分けする
df['SibSp'] = pd.Categorical(df['SibSp'], categories = all['SibSp'].unique())
df['Parch'] = pd.Categorical(df['Parch'], categories = all['Parch'].unique())
test['SibSp'] = pd.Categorical(test['SibSp'], categories = all['SibSp'].unique())
test['Parch'] = pd.Categorical(test['Parch'], categories = all['Parch'].unique())

# one-hotエンコーディングする
df = pd.get_dummies(df, columns = ['SibSp'])
df = pd.get_dummies(df, columns = ['Parch'])
test = pd.get_dummies(test, columns = ['SibSp'])
test = pd.get_dummies(test, columns = ['Parch'])

# Ticket列の各値の出現回数をカウント
ticket_values = all['Ticket'].value_counts()
# 1回しか出現していないものを省く
ticket_values = ticket_values[ticket_values > 1]
# Seriesオブジェクトに変換
ticket_values = pd.Series(ticket_values.index, name='Ticket')
# カテゴリ分けする
df['Ticket'] = pd.Categorical(df['Ticket'], categories = ticket_values.tolist())
test['Ticket'] = pd.Categorical(test['Ticket'], categories = ticket_values.tolist())
# one-hotエンコーディングする
df = pd.get_dummies(df, columns=['Ticket'])
test = pd.get_dummies(test, columns=['Ticket'])


# StandardScalerのインスタンスを作成
standard = StandardScaler()
# 標準化してDataFrameに格納
tmp = pd.DataFrame(standard.fit_transform(df[['Pclass', 'Fare']].values), columns=['Pclass', 'Fare'])
# dfに格納
df.loc[:,'Pclass'] , df.loc[:,'Fare'] = tmp['Pclass'], tmp['Fare']
# 標準化してDataFrameに格納
tmp = pd.DataFrame(standard.transform(test[['Pclass', 'Fare']].values), columns=['Pclass', 'Fare'])
# dfに格納
test.loc[:,'Pclass'], test.loc[:,'Fare'] = tmp['Pclass'], tmp['Fare']

# testの欠損値を中央値で埋める
test['Fare'].fillna(test['Fare'].median(), inplace = True)

X = df.drop(columns='Survived')
y = df[['Survived']]


kf = KFold(n_splits = 40)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    activation, out_dim, dropout = 'relu', '702', 0.5
    # モデル指定
    model = Sequential()

    # 入力層 - 隠れ層1
    model.add(Dense(input_dim = len(test.columns), units = out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層1 - 隠れ層2
    model.add(Dense(units = out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層2 - 隠れ層3
    model.add(Dense(units = out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層3 - 出力層
    model.add(Dense(units = 1))
    model.add(Activation("sigmoid"))

    # Kerasモデルをコンパイル
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',  metrics = ['accuracy'])

    # 学習
    fit = model.fit(X_train, y_train, epochs = 6, validation_data=(X_test, y_test), batch_size = 16, verbose = 2)

    # 予測
    y_test_proba = model.predict(test)
    y_test = np.round(y_test_proba).astype(int)

# PassengerId のDataFrameと結果を結合する
df_output = pd.concat([index, pd.DataFrame(y_test, columns = ['Survived'])], axis = 1)

# result.csvを書き込む
df_output.to_csv('result.csv', index = False)

# 学習データの精度を確認
accuracy = fit.history['accuracy']
loss = fit.history['loss']
val_accuracy = fit.history['val_accuracy']
val_loss = fit.history['val_loss']

# 精度の履歴をプロット
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.legend(["学習データ", "検証データ"])
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.savefig("figure/TrainingAccuracy.png")
plt.clf()

# 損失の履歴をプロット
plt.plot(loss)
plt.plot(val_loss)
plt.legend(["学習データ", "検証データ"])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.savefig("figure/Training Loss.png")
plt.clf()
