import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

'''"C:\LTSM dataset\LSTM-Bitcoin-GoogleTrends-Prediction-master\Bitcoin1D.csv"'''


# converting the data in 1D
def new_dataset(dataset, step_size):
    data_x, data_y = [], []
    for i in range(len(dataset) - step_size - 1):
        a = dataset[i:(i + step_size), 0]
        data_x.append(a)
        data_y.append(dataset[i + step_size, 0])
    return np.array(data_x), np.array(data_y)


# load DATASET
df = pd.read_csv('C:\\LTSM dataset\\LSTM-Bitcoin-GoogleTrends-Prediction-master\\Bitcoin1D.csv')
# coverting the date
df['Date'] = pd.to_datetime(df['Date'])
df = df.reindex(index=df.index[::-1])

zaman = np.arange(1, len(df) + 1, 1)
OHCL_avg = df.mean(axis=1, numeric_only=True)
plt.plot(zaman, OHCL_avg)
plt.show()
# normalise the dataset
OHCL_avg = np.reshape(OHCL_avg.values, (len(OHCL_avg), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
OHCL_avg = scaler.fit_transform(OHCL_avg)
print('done1')

# train test split
train_OHLC = int(len(OHCL_avg) * 0.56)
test_OHLC = len(OHCL_avg) - train_OHLC
train_OHLC, test_OHLC = OHCL_avg[0:train_OHLC, :], OHCL_avg[train_OHLC:len(OHCL_avg), :]
print('done 2')
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1
# ltsm model
model = Sequential()
model.add(LSTM(128, input_shape=(1, step_size)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=25, verbose=2)
trainpredict = model.predict(trainX)
testpredict = model.predict(testX)

# DE=noramalizing for ploting
trainpredict = scaler.inverse_transform(trainpredict)
trainY = scaler.inverse_transform([trainY])
testpredict = scaler.inverse_transform(testpredict)
testY = scaler.inverse_transform([testY])
# performacw measure rmse
trainscore = math.sqrt(mean_squared_error(trainY[0], trainpredict[:, 0]))
print('test RMse:%2f' % (trainscore))
# onverting plain data set for ploting
trainpredictplot = np.empty_like(OHCL_avg)
trainpredictplot[:, :] = np.nan
trainpredictplot[step_size:len(trainpredict) + step_size, :] = trainpredict

# convert predicted  test datset for plottig
testPredictPlot = np.empty_like(OHCL_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainpredict) + (step_size * 2) + 1:len(OHCL_avg) - 1, :] = testpredict
# finally predicted valuesare visualised
OHCL_avg = scaler.inverse_transform(OHCL_avg)
OHCL_avg = scaler.inverse_transform(OHCL_avg)
plt.plot(OHCL_avg, 'g', label='Orginal Dataset')
plt.plot(trainpredictplot, 'r', label='Training Set')
plt.plot(testPredictPlot, 'b', label='Predicted price/test set')
plt.title("Hourly Bitcoin Predicted Prices")
plt.xlabel('Hourly Time', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(loc='upper right')
plt.show()
