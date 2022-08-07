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
a = []
b = []

for i in range(len(pitch) - 1):
    a.append(pitch[i + 1] - pitch[i])
    b.append(pitch_rate[i + 1] - pitch_rate[i])

pitch_output = np.array(a)
pitch_rate_output = np.array(b)

input = np.stack((u_p[0:5720], pitch[0:5720], pitch_rate[0:5720]), axis=1)
output = np.stack((pitch_output[0:5720], pitch_rate_output[0:5720]), axis=1)
#output = np.stack((pitch[1:5721], pitch_rate[1:5721]), axis=1)


#print(len(input))
#print(len(output))
'''create windows'''
#print(len(input))
#print(len(output))
'''shuffle'''
train = np.concatenate((input, output), axis=1)
np.random.shuffle(train)
#print('train: ', train)
x_train = train[:4336, 0:3]
y_train = train[:4336, 3:5]
x_test = train[4336:, 0:3]
y_test = train[4336:, 3:5]
'''
x_train = train[:4336, 0:3]
y_train = train[:4336, 3:5]
x_test = train[4568:, 0:3]
y_test = train[4568:, 3:5]
'''
#print('x: ', x_test)
#print('y: ', y_test)

'''Model'''
model = Sequential()
model.add(Dense(8, input_dim=3, activation='linear'))
model.add(Dense(16, activation='linear'))
model.add(Dense(8, activation='linear'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='linear'))
#model.add(Dense(10))
#model.add(Dense(8))
#model.add(Dense(2))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()
'''Training'''
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=50)

scores = model.evaluate(x_test, y_test)
print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
#pred = model.predict(x_test)
'''save model'''
model.save('model_linear.h5')

