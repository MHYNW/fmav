import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation


data = np.array(pd.read_excel('./fmav_pitch_data.xlsx'))

time = data[:, 0]       #len = 5722
cmd = data[:, 1]
pitch = data[:, 2]
pitch_rate = data[:, 3]
u_p = data[:, 4]

input = np.stack((u_p[0:5720], pitch[0:5720], pitch_rate[0:5720]), axis=1)
output = np.stack((pitch[1:5721], pitch_rate[1:5721]), axis=1)
#print(len(input))
#print(len(output))
'''create windows'''
train__ = np.concatenate((input, output), axis=1)
lenn = 10
length = 10
result = []
for i in range(len(train__) - length):
    result.append(train__[i:i + length, :])

train = np.array(result)
np.random.shuffle(train)
#print('train: ', train)
x_train = train[:4336, :-1, 0:3]
y_train = train[:4336, -1, 3:5]
x_test = train[4336:, :-1, 0:3]
y_test = train[4336:, -1, 3:5]
#print('x: ', x_train)
#print('y: ', y_train)
#print(np.shape(x_train))
'''Model'''
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(9, 3)))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(2, activation='linear'))



#model.add(Dense(8, input_dim=3, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(4, activation='relu'))#model.add(Dense(2, activation='relu'))
#model.add(Dense(10))
#model.add(Dense(8))
#model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()
'''Training'''
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=30)

scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
#pred = model.predict(x_test)

'''save model'''
model.save('model1(01s).h5')
