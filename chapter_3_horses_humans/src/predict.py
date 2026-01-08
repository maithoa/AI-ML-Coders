import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from pathlib import Path

def load_and_predict(model_path, img_path):
    #. Check if model and image paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 2. Load model
    model = tf.keras.models.load_model(model_path)
    
    # 3. Image preprocessing
    img = image.load_img(img_path, target_size=(300, 300))
    x = image.img_to_array(img)     # <class 'numpy.ndarray'> shape (300, 300, 3)
    x /= 255.0                      # (Rescale)
    x = np.expand_dims(x, axis=0)   # Add batch batch (1, 300, 300, 3)

    # 4. Predict
    classes = model.predict(x)

    # 5. Read the result
    # With Binary Classification: 0 is Horse, 1 is Human 
    print(f"\nPrediction value (Raw Score): {classes[0][0]:.4f}")
    
    if classes[0][0] > 0.5:
        print(f"Predict: Human👤")
    else:
        print(f"Predict: Horse 🐎")

if __name__ == "__main__":
    
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    MODEL_PATH = project_root / 'saved_models' / 'horse_human_classifier_model.keras'

    # test image path
    TEST_IMAGE_PATH_HORSE = 'test_horse.jpg' 
    TEST_IMAGE_PATH_HUMAN = 'test_human.jpg' 
    TEST_IMAGE_PATH_HUMAN_2 = 'test_human_2.jpg' 
    TEST_IMAGE_PATH_HUMAN_3 = 'test_human_3.jpg' 
    TEST_IMAGE_HUMAN_AND_HORSE = 'test_human_and_horse.jpg'
    
    print("\n--- Predicting : This should be a horse ---")
    load_and_predict(MODEL_PATH, TEST_IMAGE_PATH_HORSE)

    print("\n--- Predicting : This should be a human ---")
    load_and_predict(MODEL_PATH, TEST_IMAGE_PATH_HUMAN)

    print("\n--- Predicting : This should be a human again ---")
    load_and_predict(MODEL_PATH, TEST_IMAGE_PATH_HUMAN_2)

    print("\n--- Predicting : This should be a human again 3 ---")
    load_and_predict(MODEL_PATH, TEST_IMAGE_PATH_HUMAN_3)

    print("\n--- Predicting : This can be human or horse ---")
    load_and_predict(MODEL_PATH, TEST_IMAGE_HUMAN_AND_HORSE)

# TODO: Current model relies heavily on geometric silhouettes (legs).
# Observation: Predicts accurately on full-body humans but fails on close-up faces.
# Solution: Need Data Augmentation (Ch. 4) to force facial feature learning.