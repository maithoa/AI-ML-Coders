# Get Started with TensorFlow - First Model
# This is an example script that we train a simpe model to recognise relationships between two set of numbers X and Y.
# X = –1, 0, 1, 2, 3, 4
# Y = –3, –1, 1, 3, 5, 7

import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

test_data = np.array([[10.0]], dtype=float)
print(model.predict(test_data))
print("Here is what I learned: {}".format(l0.get_weights()))

# This should output a value close to 19.0 since the relationship is Y = 2X - 1

# I wonder if I don't have to use computer resources to train model everytime I want to change the code?
