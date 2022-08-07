import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation


data = np.array(pd.read_excel('./new_data_by_steven.xlsx'))

#time = data[:, 0]       #len = 5722
'''data length 1132'''
cmd = data[:, 1]
pitch = data[:, 2]
pitch_rate = data[:, 3]
u_p = data[:, 4]

input = np.stack((u_p[0:1130], pitch[0:1130], pitch_rate[0:1130]), axis=1)
output = np.stack((pitch[1:1131], pitch_rate[1:1131]), axis=1)
#print(len(input))
#print(len(output))
'''create windows'''
train__ = np.concatenate((input, output), axis=1)
lenn = 100
length = 100
result = []
for i in range(len(train__) - length):
    result.append(train__[i:i + length, :])

train = np.array(result)
#np.random.shuffle(train)
#print('train: ', train)
'''
x_train = train[:4336, :-1, 0:3]
y_train = train[:4336, -1, 3:5]
x_test = train[4336:, :-1, 0:3]
y_test = train[4336:, -1, 3:5]
'''
x = train[100:400, :-1, 0:3]
y = train[100:400, -1, 3:5]
print(y)
#print('x: ', x_train)
#print('y: ', y_train)
#print(np.shape(x_train))
'''Model'''
'''
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(49, 3)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(2, activation='relu'))
'''
model = keras.models.load_model('model_new(1s).h5')
model.summary()


#model.add(Dense(8, input_dim=3, activation='relu'))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(4, activation='relu'))#model.add(Dense(2, activation='relu'))
#model.add(Dense(10))
#model.add(Dense(8))
#model.add(Dense(2))
'''
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.summary()
'''
'''Training'''
#model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=30)

#scores = model.evaluate(x_test, y_test)
#print('%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
#pred = model.predict(x_test)

'''save model'''
#model.save('model1(05s).h5')

'''Prediction'''
pred = model.predict(x)
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y[:, 0], label='True')
ax.plot(pred[:, 0], label='Prediction')
ax.plot(cmd[100:400], label='cmd')
ax.legend()
plt.show()
