import numpy as np
import pandas as pd

df = pd.read_excel("reference.xlsx", usecols="A")
reference = df.iloc[0][0]

df1 = pd.read_excel("sequence.xlsx", usecols="A")
dataset = [df1.iloc[i][0] for i in range(0,82)]

nuc = ["a", "c", "g", "t"]

mt = []
for i in range(0,len(dataset)):
    mutation = np.zeros((4,4))
    for j in range(0,min(len(reference),len(dataset[i]))):
        D1 = dataset[i][j]
        D2 = reference[j]
        if D1!=D2:
            mutation[nuc.index(D1)][nuc.index(D2)] += 1
    MutationRate = (mutation/(1*len(reference)))*100
    mt.append(MutationRate)


mt = [m[~np.eye(m.shape[0],dtype=bool)].reshape(m.shape[0],-1) for m in mt]
mt = [m.flatten() for m in mt]

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = mt[0:train_size], mt[train_size:len(dataset)]

features = list(train[i:i+12] for i in range(0,train_size-12))
labels = list(train[i] for i in range(12,train_size))

# establishing lstm
from numpy.random import seed
seed(1)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

model = Sequential()
model.add(LSTM(500))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(12, activation='relu'))
model.compile(optimizer='adam',loss='mse')

hist = model.fit(np.array(features),np.array(labels),epochs=10)

t_features = list(test[i:i+11] for i in range(0,test_size-11))
t_labels = list(test[i] for i in range(11,test_size))
model.predict(np.array(t_features))

print(np.array(t_labels))