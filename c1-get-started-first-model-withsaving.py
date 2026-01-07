import os
import numpy as np
import tensorflow as tf
from keras import Sequential, layers, models

# Configuration
# Define the folder and filename
MODEL_DIR = 'models'
MODEL_NAME = 'my_1st_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF optimization warnings

def run_prediction():
    # Check if a pre-trained model already exists
    if os.path.exists(MODEL_PATH):
        print(f"--- Loading existing model from {MODEL_PATH} ---")
        model = models.load_model(MODEL_PATH)
    else:
        print("--- No saved model found. Starting training... ---")
        
        l0 = layers.Dense(units=1, input_shape=[1])
        # Define the architecture using Keras 3 standards
        model = Sequential([
            l0
        ])
        
        model.compile(optimizer='sgd', loss='mean_squared_error')

        # Training data
        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

        # Train the model
        model.fit(xs, ys, epochs=500, verbose=0) # verbose=0 to keep terminal clean
        
        # Save the model to avoid re-training next time
        model.save(MODEL_PATH)
        print(f"--- Training complete. Model saved to {MODEL_PATH} ---")

    # Perform prediction
    # Note: Using 2D numpy array to comply with Keras 3 requirements
    test_input = np.array([[10.0]], dtype=float)
    prediction = model.predict(test_input)
    
    print(f"Prediction for input 10.0: {prediction}")


if __name__ == "__main__":
    run_prediction()