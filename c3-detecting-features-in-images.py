import tensorflow as tf
data=tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_iamges, tests_labels) = data.load_data()

training_images=training_images.reshape(60000,28,28,1)
test_iamges=test_iamges.reshape(10000,28,28,1)

# Normalize the images to a range of 0 to 1
training_images = training_images / 255.0
test_iamges = test_iamges / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_iamges, tests_labels)

classifications = model.predict(test_iamges)
print('\nPrediction for first test image:', classifications[0])
print('\nLabel for first test image:', tests_labels[0])

print("\nHere is what I learned: {}".format(model.summary()))



