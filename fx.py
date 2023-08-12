#ライブラリのインポート
import csv
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.foreignexchange import ForeignExchange
from datetime import datetime, timedelta

# Alpha Vantage APIキー
api_key = 'YOUR API KEY'

# USDJPYの為替データを取得
fx = ForeignExchange(key=api_key)
data, _ = fx.get_currency_exchange_daily('USD', 'JPY', outputsize='full')

# グラフ表示用
dates = []
open_values = []
high_values = []
low_values = []
close_values = []

#データセット用
fx_data = []

i = 0

# データの解析
for date, values in data.items():
    if i < 1000:
        column = []
        
        #確認用
        print("Date:", date)
        print("Open:", values['1. open'])
        print("High:", values['2. high'])
        print("Low:", values['3. low'])
        print("Close:", values['4. close'])
        print()

        #グラフ用
        dates.append(date)
        open_values.append(float(values['1. open']))
        high_values.append(float(values['2. high']))
        low_values.append(float(values['3. low']))
        close_values.append(float(values['4. close']))
        
        #データセット用
        column.append(date)
        column.append(float(values['1. open']))
        column.append(float(values['2. high']))
        column.append(float(values['3. low']))
        column.append(float(values['4. close']))
        fx_data.append(column)

        i += 1
    else:
        #データセット用
        column = []
        column.append(date)
        column.append(float(values['1. open']))
        column.append(float(values['2. high']))
        column.append(float(values['3. low']))
        column.append(float(values['4. close']))
        fx_data.append(column)            

#データを逆順にする
dates.reverse()
open_values.reverse()
high_values.reverse()
low_values.reverse()
close_values.reverse()
fx_data.reverse()

print(type(data))
print(data)

#データフレーム作成
fx_df = pd.DataFrame(fx_data)
fx_df = fx_df.set_axis(['Datetime', 'open', 'high', 'low', 'close'], axis=1)


plt.figure(figsize=(15,4))
plt.style.use('ggplot')
plt.rcParams["font.size"] = 8

plt.plot(dates, open_values, label='Open Values', linewidth=1)
plt.plot(dates, high_values, label='High Values', linewidth=1)
plt.plot(dates, low_values, label='Low Values', linewidth=1)
plt.plot(dates, close_values, label='Close Values', linewidth=2)

plt.title('USDJPY Exchange Rate')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)  # X軸の日付を回転して表示
plt.xticks(dates[::100])
plt.legend()
plt.tight_layout()
plt.show()


#必要なライブラリのインポート
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime

#fx_dfから終値だけのデータフレームを作成する。（インデックスは残っている）
rdata = fx_df.filter(['close']) 
#dataをデータフレームから配列に変換する。
dataset = rdata.values

#トレーニングデータを格納するための変数を作る。
#データの80%をトレーニングデータとして使用する。
#math.ceil()は小数点以下の切り上げ
training_data_len = math.ceil(len(dataset) * 0.8)

#データセットを0から1までの間にスケーリング
scaler = MinMaxScaler(feature_range=(0,1))
#fitは変換式を計算する
#transformはfitの結果を使って、実際にデータを変換する
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0: training_data_len, :]

#データをx_trainとy_trainのセットに分ける
x_train =[]
y_train =[]
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

#LSTMに受け入れられる形にデータを作り変える
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTMモデルを構築。50ニューロンの2つのLSTMレイヤーと2つの高密度レイヤーを作成
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#平均二乗誤差（MSE）と損失関数とadamオプティマイザーを使用してモデルをコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

#訓練の実行
model.fit(x_train, y_train, batch_size=1, epochs=1)

#test data set
test_data = scaled_data[training_data_len - 60:, : ]

#create the x_test and y_test data sets
x_test = [] #予測値を作るために使用するデータ
y_test = dataset[training_data_len : , : ] #実際の終値データ

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0]) #正規化データ

#convert x_test to a numpy array
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#テストデータを使用してモデルから予測値を取得
#Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #Undo scaling

#Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse

#Let’s plot and visualize the data.
#Plot/Create the data for the graph
train = rdata[:training_data_len]
valid = rdata[training_data_len:]
valid['Predictions'] = predictions #予測値のデータ

model.summary() #モデル情報を出力


# 'fx_df' から 'close' 列だけを抽出して 'new_df' を作成
new_df = fx_df.filter(['close'])

# 最後の60日間のデータを取得
last_60_days = new_df[-60:].values

predict_array = np.array([[0]])

super_predictions_len = 1000

for i in range(super_predictions_len):
    # データを0から1の範囲にスケーリング
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    # 過去60日分のデータを 'X_test' に追加
    X_test.append(last_60_days_scaled)
    # 'X_test' を numpy 配列に変換
    X_test = np.array(X_test)
    # データの形状を変更
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # モデルを使って価格を予測
    pred_price = model.predict(X_test)

    # スケーリングを元に戻す
    pred_price = scaler.inverse_transform(pred_price)

    #last_60_daysの最初の行を削除、最後の行に追加
    last_60_days = np.delete(last_60_days, 0, 0)
    last_60_days = np.append(last_60_days, pred_price, axis=0)
    
    predict_array = np.append(predict_array, pred_price, axis=0)

predict_array = np.delete(predict_array, 0, 0)

super_predictions = pd.DataFrame(
    data = predict_array,
    index = list(range(5000, 5000 + super_predictions_len)),
    columns = ['super_predictions']
)

print(super_predictions)


plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.plot(super_predictions[['super_predictions']])
plt.legend(['Train', 'Val', 'Predictions', 'super_predictions'], loc='lower left')
plt.show()
plt.savefig('model.png')


new_df = fx_df.filter(['close']) 

last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)