from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# データセットを取得
housing = fetch_california_housing(as_frame=True)

# DataTableのheaderをlist化
header = list(housing["frame"].columns.values)
allList = [housing["frame"][i].to_list() for i in header]
''' ヘッダー名と説明
0 MedInc（median income）： 各ブロックグループ内にある世帯ごとの「所得」の中央値。※明示されていないが、1990年のカリフォルニアの世帯収入の中央値が3万ドル台であるため、恐らく1万ドル（＝10,000ドル）単位だと推定される
1 HouseAge（median house age）： ブロックグループの「築年数」の中央値
2 AveRooms（avarage number of rooms）： 各ブロックグループ内にある世帯ごとの「部屋数」の平均値（＝1世帯当たりの部屋数。※元は総部屋数）
3 AveBedrms（avarage number of bedrooms）： 各ブロックグループ内にある世帯ごとの「寝室数」の平均値（＝1世帯当たりの寝室数。※元は総寝室数）
4 Population： ブロックグループの「人口」（＝居住する人の総数）
5 AveOccup（average occupancy rate）： 各ブロックグループ内にある世帯ごとの「世帯人数」の平均値（＝1世帯当たりの世帯人数。※元は総世帯数）
6 Latitude： ブロックグループの中心点の「緯度」。値が＋方向に大きいほど、そのブロックグループは北にある
7 Longitude： ブロックグループの中心点の「経度」。値が－方向に大きいほど、そのブロックグループは西にある
8 MedHouseVal （median house value）：「住宅価格」（100,000ドル＝10万ドル単位）の中央値。通常はこの数値が目的変数として使われる
'''
# 辞書作成
titleDict = {
    "MedInc" : "所得の中央値",
    "HouseAge" : "築年数の中央値",
    "AveRooms" : "部屋数の平均値",
    "AveBedrms" : "寝室数の平均値",
    "Population" : "人口",
    "AveOccup" : "世帯人数の平均値",
    "Latitude" : "ブロック中心の緯度",
    "Longitude" : "ブロック中心の経度",
    "MedHouseVal" : "住宅価格の中央値(10万ドル単位)"
}

def ex(header, allList, titleDict, X, Y):
    # フォルダ作成
    os.makedirs("figure", exist_ok=True)

    # 配列定義
    x = np.array(allList[X])
    x = x.reshape(-1, 1)
    y = np.array(allList[Y])

    # 予測式を求める
    model = LinearRegression()
    model.fit(x, y)

    # サンプルを生成
    qty = len(allList[0])*random.uniform(0.05, 0.1)
    source = [allList[X][i] for i in random.sample(range(0, len(allList[0])), int(qty))]
    source.sort()
    source = [[i] for i in source]

    # 予測
    predicted = model.predict(source)

    # プロット
    title = f"{titleDict[header[X]]} - {titleDict[header[Y]]}" # タイトル作成
    plt.title(title) # タイトル表示
    plt.xlim(min(allList[X]), max(allList[X])) # データの最小値を最大値をグラフの最大値と最小値に
    plt.ylim(min(allList[X]), max(allList[Y]))
    plt.scatter(x, y) # プロット
    plt.xlabel(titleDict[header[X]]) # X軸ラベル
    plt.ylabel(titleDict[header[Y]]) # Y軸ラベル
    plt.plot(source, predicted, color = 'red') # 予測線
    plt.savefig(f"figure/{header[X]}-{header[Y]}.png") # エクスポート
    plt.clf() # プロットクリア

    # json形式でも結果を保存
    tmp = {
    "関数":f"y = {model.coef_[0]}x + {model.intercept_}",
    "x" : f"{source}",
    "y" : f"{predicted}"
    }
    return [title, tmp]

# 全部について最小二乗法を行う
rst = {ex(header, allList, titleDict, i, j)[0] : ex(header, allList, titleDict, i, j)[1] for i in range(len(header)) for j in range(i+1, len(header))}

# json出力
with open('result.json', 'w') as f:
    json.dump(rst, f, ensure_ascii=False, indent=4)