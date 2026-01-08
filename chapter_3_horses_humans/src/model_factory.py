import tensorflow as tf
from keras import layers, models

def create_model(input_shape_param=(300, 300, 3)):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape_param),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model