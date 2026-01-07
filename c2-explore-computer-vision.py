import tensorflow as tf
data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model =tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 

model.fit(train_images, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

classifications = model.predict(test_images)
print('\nPrediction for first test image:', classifications[0])
print('\nLabel for first test image:', test_labels[0])

