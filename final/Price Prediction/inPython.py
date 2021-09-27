from IPython.core.debugger import set_trace

# %load_ext nb_black

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time

plt.style.use(style="seaborn")
#%matplotlib inline
#df = pd.read_csv("data/MSFT-1Y-Hourly.csv")
df=pd.read_csv("data/merged.csv")
df.set_index("date", drop=True, inplace=True)
df = df[["close"]]
df.describe()
df["returns"] = df.close.pct_change()
df["log_returns"] = np.log(1 + df["returns"])
df.dropna(inplace=True) #drop NANs
X = df[["close", "log_returns"]].values
X_backup=X

#TODO testar outros escaladores
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
# scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = [x[0] for x in X]
y_backup = [x[0] for x in X_backup]


from sklearn.preprocessing import MinMaxScaler, StandardScaler
split = int(len(X) * 0.8)
X_train = X[:split]
X_test = X[split : len(X)]
y_train = y[:split]
y_test = y[split : len(y)]
y_train_backup = y_backup[:split]
y_test_backup = y_backup[split : len(y)]

#TODO laco para varios tamanhos de janelas
n = 7
Xtrain = []
ytrain = []
Xtest = []
ytest = []
for i in range(n, len(X_train)):
    Xtrain.append(X_train[i - n : i, : X_train.shape[1]])
    ytrain.append(y_train[i])  # predict next record
for i in range(n, len(X_test)):
    Xtest.append(X_test[i - n : i, : X_test.shape[1]])
    ytest.append(y_test[i])  # predict next record

#Therefore we need to add a temporal dimension compared to a classical network:
#(number of observations, number of steps, number of features per step)
Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))
Xtest, ytest = (np.array(Xtest), np.array(ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))


#################################################################
##################### LSTM MODEL ################################
#################################################################
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_loss', patience=50)
model = Sequential()
model.add(LSTM(100, input_shape=(Xtrain.shape[1], Xtrain.shape[2]),return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(100,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(100,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(20,return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer=Adam(1e-3))
history=model.fit(
    Xtrain, ytrain, epochs=250, validation_data=(Xtest, ytest), 
    batch_size=16, verbose=1, callbacks=[callback]
)

trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)
trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]

trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = [x[0] for x in trainPredict]
testPredict = scaler.inverse_transform(testPredict)
testPredict = [x[0] for x in testPredict]

#LEARNING PLOT
fig, ax1 = plt.subplots(constrained_layout=True)
ax2 = ax1.twinx()
ax1.plot(history.history['loss'], label='train')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.plot(history.history['val_loss'],color='green', label='val')
ax2.set_ylabel('val loss')
plt.show()
print(model.summary())

#CSV OUTPUT
real_test=y_test_backup
print("real_test",len(real_test),real_test[:10])
d = {"real":real_test,"pred":(list(np.zeros(n))+testPredict)}
out=pd.DataFrame(data=d)
out.to_csv("vamoa.csv")

#NOT GOOD PLOT
print(range(len(trainPredict)+n,len(trainPredict)+len(testPredict)+n))
fig, ax = plt.subplots(figsize=(15,10))
plt.plot(df.close,color='red', label="real")
plt.plot(list(np.zeros(n))+(trainPredict),color='green', label="Predicted Train")
ax.plot(range(len(trainPredict)+n,
              len(trainPredict)+len(testPredict)+n),
              testPredict,
              color='blue',
              label='Predicted Test')
plt.legend()
#TODO so o teste


from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
real_train=y_train_backup
# print(real_train[n:,0].shape,len(trainPredict))
# print(real_train[n:,0][:5])
trainScore = mean_squared_error(real_train[n:], trainPredict, squared=False)
print("train Score: %.2f RMSE" % (trainScore))
testScore = mean_squared_error(real_test[n:], testPredict, squared=False)
print("Test Score: %.2f RMSE" % (testScore))

#UNKNOWN PAIR
df2=pd.read_csv("data/eth.csv")
df2.set_index("date", drop=True, inplace=True)
df2 = df2[["close"]]
df2["returns"] = df2.close.pct_change()
df2["log_returns"] = np.log(1 + df2["returns"])
df2.dropna(inplace=True) #drop NANs
X2 = df2[["close", "log_returns"]].values
X_backup2=X2
print(X2.shape)
scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(X2)
X2 = scaler2.transform(X2)
y2 = [x[0] for x in X2]
y_backup2 = [x[0] for x in X_backup2]
Xtest2 = []
ytest2 = []
for i in range(n, len(X2)):
    Xtest2.append(X2[i - n : i, : X2.shape[1]])
    ytest2.append(y2[i])  # predict next record
Xtest2, ytest2 = (np.array(Xtest2), np.array(ytest2))
Xtest2 = np.reshape(Xtest2, (Xtest2.shape[0], Xtest2.shape[1], Xtest2.shape[2]))
print(Xtest2.shape)
Predict2 = model.predict(Xtest2)
Predict2 = np.c_[Predict2, np.zeros(Predict2.shape)]
Predict2 = scaler2.inverse_transform(Predict2)
Predict2 = [x[0] for x in Predict2]
# real_test2 = np.c_[ytest2, np.zeros(len(ytest2))]
# real_test2 = scaler2.inverse_transform(real_test2)
# real_test2 = real_test2.T.tolist()[0]
real_test2=y_backup2
print(type(real_test2),len(real_test2),type(Predict2),len(Predict2))
# print("targetY",len(scaler2.inverse_transform(real_test2)),real_test2[:10])
d2 = {"real":real_test2[n:],"pred":Predict2}
out2=pd.DataFrame(data=d2)
out2.to_csv("vamoa2.csv")
testScore = mean_squared_error(real_test2[n:], Predict2, squared=False)
print("Test Score: %.2f RMSE" % (testScore))