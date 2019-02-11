import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
'''
print(x_train)
plt.imshow(x_train[0])
print(x_train[0])
plt.show()
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#-----------Predicting the result--------------------
predictions = model.predict([x_test])

import numpy as np
print(np.argmax(predictions[5]))
plt.imshow(x_test[5])
plt.show()
