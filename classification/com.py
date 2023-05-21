from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Activation, BatchNormalization, Dense
from keras.layers.core import Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

e = 50
accuracy = []
loss = []
val_accuracy = []
val_loss = []

for i in range(4):
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


    match i:
        case 0: # PclassごとのAgeの平均値で埋める
            df['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.mean()))['Age']
            test['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.mean()))['Age']
        case 1: # PclassごとのAgeの中央値で埋める
            df['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.median()))['Age']
            test['Age'] = df.groupby('Pclass').transform(lambda x: x.fillna(x.median()))['Age']
        case 2: # 平均値で埋める
            df['Age'] = df['Age'].fillna(df['Age'].mean())
            test['Age'] = test['Age'].fillna(test['Age'].mean())
        case 3: # 中央値で埋める
            df['Age'] = df['Age'].fillna(df['Age'].median())
            test['Age'] = test['Age'].fillna(test['Age'].median())

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

    # 学習データと検証データを分ける
    df, vali = train_test_split(df,test_size=0.4,random_state=0)

    # 学習データ
    x = df.drop(columns='Survived')
    y = df[['Survived']]

    # 検証データ
    vx = vali.drop(columns='Survived')
    vy = vali[['Survived']]

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
    fit = model.fit(x, y, validation_data=(vx, vy), epochs = e, batch_size = 16, verbose = 2)

    # 予測
    y_test_proba = model.predict(test)
    y_test = np.round(y_test_proba).astype(int)

    # PassengerId のDataFrameと結果を結合する
    df_output = pd.concat([index, pd.DataFrame(y_test, columns = ['Survived'])], axis = 1)

    # 学習データの精度を確認
    accuracy.append(fit.history['accuracy'])
    loss.append(fit.history['loss'])
    val_accuracy.append(fit.history['val_accuracy'])
    val_loss.append(fit.history['val_loss'])

# 精度の履歴をプロット
for i in range(len(accuracy)):
    plt.plot(accuracy[i])
plt.title('学習精度')
plt.xlabel('Epochs')
plt.legend(["PclassごとのAgeの平均値補完", "PclassごとのAgeの中央値補完" ,"平均値", "中央値"])
plt.savefig("figure/ComparisonTrainingAccuracy.png")
plt.clf()

# 損失の履歴をプロット
for i in range(len(loss)):
    plt.plot(loss[i])
plt.title('学習損失')
plt.xlabel('Epochs')
plt.legend(["PclassごとのAgeの平均値補完", "PclassごとのAgeの中央値補完" ,"平均値", "中央値"])
plt.savefig("figure/ComparisonTrainingLoss.png")
plt.clf()